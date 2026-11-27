"""Запуск: python -m validator <файл требований> <файл отчёта> [опции]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import __version__, document, report
from .llm import LLMClient, LLMError
from .pipeline import Agent, dumps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m validator",
        description="ИИ-агент статической проверки требований (ISO/IEC/IEEE 29148:2018, IEEE 830-1998).")
    ap.add_argument("input", help="путь к документу требований (PDF, DOCX, MD, TXT, HTML)")
    ap.add_argument("output", help="путь к файлу отчёта (.md); рядом ляжет .json с теми же данными")
    ap.add_argument("--provider", choices=["gemini", "openai", "anthropic", "ollama"],
                    help="провайдер LLM (по умолчанию — по ключу в .env)")
    ap.add_argument("--model", help="имя модели (по умолчанию из .env)")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="после анализа задать уточняющие вопросы в терминале и учесть ответы")
    ap.add_argument("--answers", help="JSON-файл с ответами на вопросы {\"Q1\": \"…\"} (неинтерактивно)")
    ap.add_argument("--no-critic", action="store_true", help="отключить проход-рецензент")
    ap.add_argument("--no-cache", action="store_true", help="не использовать кэш ответов модели")
    ap.add_argument("--env", default=".env", help="путь к .env (по умолчанию ./.env)")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    a = ap.parse_args(argv)

    load_dotenv(a.env)
    log = (lambda *x: None) if a.quiet else (lambda *x: print(*x, flush=True))

    try:
        doc = document.load(a.input)
        llm = LLMClient(provider=a.provider, model=a.model, cache_dir=None if a.no_cache else ".cache",
                        verbose=not a.quiet)
    except (FileNotFoundError, ValueError, LLMError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 2

    log(f"Документ: {doc.path.name} ({doc.fmt}, {len(doc.lines)} строк). Модель: {llm.label}")
    agent = Agent(llm, use_critic=not a.no_critic, log=log)
    try:
        res = agent.run(doc)
        answers = {}
        if a.answers:
            answers = json.loads(Path(a.answers).read_text("utf-8"))
        elif a.interactive and res.questions:
            answers = ask(res.questions)
            if answers:
                p = Path(a.output).with_suffix(".answers.json")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(answers, ensure_ascii=False, indent=2), "utf-8")
                log(f"Ответы сохранены: {p} (повтор прогона: --answers {p})")
        if answers:
            log("▶ 8. Учёт ответов на уточняющие вопросы")
            agent.clarify(res, answers)
    except LLMError as e:
        print(f"Ошибка обращения к модели: {e}", file=sys.stderr)
        return 3

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.render(res, llm), "utf-8")
    out.with_suffix(".json").write_text(dumps(res), "utf-8")
    sev = {s: sum(1 for f in res.findings if f["severity"] == s) for s in ("critical", "major", "minor")}
    log(f"\nГотово. Требований: {len(res.requirements)}, дефектов: {len(res.findings)} "
        f"(критичных {sev['critical']}, существенных {sev['major']}, незначительных {sev['minor']}).")
    log(f"Отчёт: {out}\nДанные: {out.with_suffix('.json')}")
    return 0


def ask(questions) -> dict:
    print("\n" + "=" * 70)
    print("Агенту нужны уточнения. Ответьте на вопросы (Enter — пропустить, q — закончить).")
    print("=" * 70)
    answers = {}
    for q in questions:
        print(f"\n{q.get('id')}. [{q.get('to', '')}] {q.get('question')}")
        if q.get("options"):
            for i, o in enumerate(q["options"], 1):
                print(f"   {i}) {o}")
        if q.get("default"):
            print(f"   Предложение агента: {q['default']}  (введите «+», чтобы принять)")
        try:
            ans = input("   Ответ: ").strip()
        except EOFError:
            break
        if ans.lower() == "q":
            break
        if ans == "+" and q.get("default"):
            ans = q["default"]
        elif ans.isdigit() and q.get("options") and 1 <= int(ans) <= len(q["options"]):
            ans = q["options"][int(ans) - 1]
        if ans:
            answers[q["id"]] = ans
    return answers


if __name__ == "__main__":
    sys.exit(main())
