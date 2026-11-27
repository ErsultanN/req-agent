"""Тонкий клиент к LLM: Gemini (по умолчанию), OpenAI-совместимые API (OpenAI, Ollama, vLLM,
OpenRouter, LM Studio), Anthropic. Только HTTP через requests — без тяжёлых SDK.

Возможности: ответ строго в JSON, повторы при 429/5xx с экспоненциальной паузой,
ограничение частоты запросов (бесплатный тариф), дисковый кэш для воспроизводимости.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests


class LLMError(RuntimeError):
    pass


@dataclass
class LLMStats:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    log: list = field(default_factory=list)


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 temperature: float | None = None, cache_dir: str | None = ".cache",
                 min_interval: float | None = None, timeout: int = 180, verbose: bool = False):
        self.provider = (provider or os.getenv("LLM_PROVIDER") or _autodetect()).lower()
        self.model = model or _default_model(self.provider)
        if temperature is None:
            env_t = os.getenv("LLM_TEMPERATURE")
            # Google рекомендует для Gemini 3.x температуру по умолчанию 1.0; для остальных — 0 (воспроизводимость)
            temperature = float(env_t) if env_t else (1.0 if self.model.startswith("gemini-3") else 0.0)
        self.temperature = temperature
        self.timeout = timeout
        self.verbose = verbose
        self.min_interval = float(min_interval if min_interval is not None
                                  else os.getenv("LLM_MIN_INTERVAL", "4" if self.provider == "gemini" else "0"))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = LLMStats()
        self._last_call = 0.0
        self._check_key()

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    # ------------------------------------------------------------------
    def json(self, system: str, user: str, stage: str = "") -> dict:
        """Запрос, ответ которого обязан быть JSON-объектом."""
        key = hashlib.sha256(json.dumps([self.label, self.temperature, system, user]).encode()).hexdigest()[:24]
        cpath = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cpath and cpath.exists():
            self.stats.cached += 1
            self.stats.log.append({"stage": stage, "cached": True})
            return json.loads(cpath.read_text("utf-8"))

        last_err = None
        attempts = int(os.getenv("LLM_RETRIES", "8"))
        for attempt in range(attempts):
            try:
                text = self._call(system, user)
                data = parse_json(text)
                if cpath:
                    cpath.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
                return data
            except LLMError as e:
                last_err = e
                msg = str(e)
                if "429" in msg and re.search(r"per ?day|PerDay|requests_per_day|daily", msg, re.I):
                    raise LLMError("Исчерпан ДНЕВНОЙ лимит запросов бесплатного тарифа для модели "
                                   f"{self.model}. Повторите после сброса квоты (полночь по тихоокеанскому "
                                   "времени, ~13:00 по Алматы) или выберите другую модель: "
                                   "--model gemini-3.5-flash-lite. Готовые ответы кэшируются, повторный "
                                   "прогон доберёт только недостающее.")
                retriable = any(s in msg for s in ("429", "500", "502", "503", "504", "timeout", "JSON", "overloaded"))
                if not retriable:
                    raise
                # 429 на бесплатном тарифе — поминутная квота: ждать нужно ~минуту, а не секунды
                wait = min(90, (30 if "429" in msg else 5) * 2 ** min(attempt, 2))
                m = re.search(r"retry in ([\d.]+)s|retryDelay\": \"(\d+)s", msg)
                if m:
                    wait = max(wait, float(m.group(1) or m.group(2)) + 1)
                short = "лимит запросов бесплатного тарифа (429)" if "429" in msg else msg[:160]
                self._say(f"  ! {stage}: {short} — повтор через {wait:.0f} с")
                time.sleep(wait)
        raise LLMError(f"LLM не ответила корректно после {attempts} попыток: {last_err}")

    # ------------------------------------------------------------------
    def _call(self, system: str, user: str) -> str:
        gap = time.time() - self._last_call
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        t0 = time.time()
        try:
            if self.provider == "gemini":
                text, tin, tout = self._gemini(system, user)
            elif self.provider == "anthropic":
                text, tin, tout = self._anthropic(system, user)
            else:
                text, tin, tout = self._openai(system, user)
        except requests.Timeout as e:
            raise LLMError(f"timeout: {e}")
        except requests.RequestException as e:
            raise LLMError(f"503 network: {e}")
        finally:
            self._last_call = time.time()
        dt = time.time() - t0
        self.stats.calls += 1
        self.stats.input_tokens += tin
        self.stats.output_tokens += tout
        self.stats.seconds += dt
        self.stats.log.append({"stage": "", "seconds": round(dt, 1), "in": tin, "out": tout})
        return text

    def _gemini(self, system, user):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": self.temperature, "responseMimeType": "application/json",
                                 "maxOutputTokens": int(os.getenv("LLM_MAX_TOKENS", "32768"))},
        }
        r = requests.post(url, json=body, timeout=self.timeout,
                          headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        if r.status_code != 200:
            raise LLMError(f"Gemini HTTP {r.status_code}: {r.text[:500]}")
        j = r.json()
        try:
            parts = j["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (KeyError, IndexError):
            raise LLMError(f"Gemini: пустой ответ (JSON?): {str(j)[:300]}")
        u = j.get("usageMetadata", {})
        return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)

    def _openai(self, system, user):
        base = (os.getenv("OPENAI_BASE_URL") or
                (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1"
                 if self.provider == "ollama" else "https://api.openai.com/v1")).rstrip("/")
        key = os.getenv("OPENAI_API_KEY", "ollama")
        body = {"model": self.model, "temperature": self.temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}}
        r = requests.post(f"{base}/chat/completions", json=body, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {key}"})
        if r.status_code != 200:
            raise LLMError(f"OpenAI-compatible HTTP {r.status_code}: {r.text[:500]}")
        j = r.json()
        u = j.get("usage", {})
        return j["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

    def _anthropic(self, system, user):
        body = {"model": self.model, "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "16000")),
                "temperature": self.temperature, "system": system,
                "messages": [{"role": "user", "content": user + "\n\nОтветь только JSON-объектом."}]}
        r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=self.timeout,
                          headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                   "anthropic-version": "2023-06-01"})
        if r.status_code != 200:
            raise LLMError(f"Anthropic HTTP {r.status_code}: {r.text[:500]}")
        j = r.json()
        text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
        u = j.get("usage", {})
        return text, u.get("input_tokens", 0), u.get("output_tokens", 0)

    def _check_key(self):
        need = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
        var = need.get(self.provider)
        if var and not os.getenv(var):
            raise LLMError(f"Не задан {var}. Скопируйте .env.example в .env и впишите ключ (см. README).")

    def _say(self, msg):
        if self.verbose:
            print(msg, flush=True)


def _autodetect() -> str:
    for var, prov in (("GEMINI_API_KEY", "gemini"), ("OPENAI_API_KEY", "openai"),
                      ("ANTHROPIC_API_KEY", "anthropic"), ("OLLAMA_MODEL", "ollama")):
        v = os.getenv(var, "")
        if v and "your_" not in v:
            return prov
    return "gemini"


def _default_model(provider: str) -> str:
    env = {"gemini": "GEMINI_MODEL", "openai": "OPENAI_MODEL", "anthropic": "ANTHROPIC_MODEL",
           "ollama": "OLLAMA_MODEL"}.get(provider, "LLM_MODEL")
    defaults = {"gemini": "gemini-3.6-flash", "openai": "gpt-4.1-mini",
                "anthropic": "claude-sonnet-5-5", "ollama": "qwen2.5:14b"}
    return os.getenv(env) or os.getenv("LLM_MODEL") or defaults.get(provider, "")


def parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        v = json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a < 0 or b <= a:
            raise LLMError(f"JSON не найден в ответе модели: {t[:200]}")
        try:
            v = json.loads(t[a:b + 1])
        except json.JSONDecodeError as e:
            raise LLMError(f"Битый JSON в ответе модели: {e}")
    if isinstance(v, list):
        v = {"items": v}
    return v
