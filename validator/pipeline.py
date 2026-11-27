"""Конвейер проверки: загрузка → линтер → нарезка → проверка требований → проверка набора →
независимый рецензент → проверка цитат → вопросы и итог → (опционально) уточнения в диалоге."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from . import prompts, rules
from .criteria import REQ_CRITERIA, SET_CRITERIA, SEVERITY_ORDER
from .document import Document
from .llm import LLMClient, LLMError


@dataclass
class RunResult:
    doc: Document
    model: str
    started: float
    finished: float = 0.0
    parts: list = field(default_factory=list)
    requirements: list = field(default_factory=list)      # нарезка
    req_verdicts: dict = field(default_factory=dict)      # rid -> {criterion: ok|issue|na}
    rewrites: dict = field(default_factory=dict)          # rid -> исправленная формулировка
    set_verdicts: dict = field(default_factory=dict)
    coverage: list = field(default_factory=list)
    findings: list = field(default_factory=list)          # итоговые подтверждённые
    rejected: list = field(default_factory=list)          # отсеянные (рецензент / цитата не найдена)
    questions: list = field(default_factory=list)
    verdict: str = "no"
    verdict_reason: str = ""
    priorities: list = field(default_factory=list)
    clarifications: dict = field(default_factory=dict)    # ответы на вопросы (диалог)
    warnings: list = field(default_factory=list)
    stage_times: dict = field(default_factory=dict)


class Agent:
    def __init__(self, llm: LLMClient, batch_size: int | None = None, use_critic: bool = True,
                 log=print):
        self.llm = llm
        self.batch_size = batch_size or int(os.getenv("AGENT_BATCH_SIZE", "8"))
        self.use_critic = use_critic
        self.log = log

    # ------------------------------------------------------------------
    def run(self, doc: Document) -> RunResult:
        res = RunResult(doc=doc, model=self.llm.label, started=time.time())
        numbered = doc.numbered_text()

        with self._stage(res, "1. Детерминированные проверки"):
            rule_findings = rules.run(doc)
            self.log(f"   правил сработало: {len(rule_findings)}")
            hints = rules.hints_text(rule_findings)

        with self._stage(res, "2. Нарезка требований"):
            seg = self.llm.json(prompts.SYSTEM, prompts.SEGMENT.format(doc=numbered), stage="segment")
            res.parts = seg.get("parts", [])
            res.requirements = self._normalize_reqs(doc, seg.get("requirements", []), res)
            if seg.get("document_title"):
                doc.title = seg["document_title"]
            self.log(f"   требований: {len(res.requirements)}")

        raw: list[dict] = []
        with self._stage(res, "3. Проверка каждого требования"):
            reqs = res.requirements
            for i in range(0, len(reqs), self.batch_size):
                batch = reqs[i:i + self.batch_size]
                self.log(f"   пакет {i // self.batch_size + 1}: {batch[0]['rid']}…{batch[-1]['rid']}")
                block = "\n".join(
                    f"- {r['rid']} (в документе: {r['doc_id'] or 'без номера'}; раздел «{r['section']}»; "
                    f"L{r['lines'][0]}-L{r['lines'][1]}): «{r['quote']}»"
                    + (f"; составляющие: {'; '.join(r['atoms'])}" if len(r.get('atoms') or []) > 1 else "")
                    for r in batch)
                rel_hints = rules.hints_text([f for f in rule_findings if any(
                    r["lines"][0] - 1 <= f["lines"][0] <= r["lines"][1] + 1 for r in batch)])
                out = self.llm.json(prompts.SYSTEM, prompts.PER_REQ.format(
                    criteria=prompts.req_criteria_block(), hints=rel_hints, reqs=block, doc=numbered),
                    stage="per_req")
                for item in out.get("results", []):
                    rid = item.get("rid")
                    if rid not in {r["rid"] for r in batch}:
                        continue
                    res.req_verdicts[rid] = {k: _verdict(item.get("verdicts", {}).get(k)) for k in REQ_CRITERIA}
                    if item.get("rewrite"):
                        res.rewrites[rid] = item["rewrite"]
                    for f in item.get("findings", []):
                        f.update(scope="requirement", source="llm")
                        f["related_rids"] = sorted(set([rid] + (f.get("related_rids") or [])))
                        raw.append(f)
            missing = [r["rid"] for r in reqs if r["rid"] not in res.req_verdicts]
            if missing:
                res.warnings.append(f"Модель не вернула оценку для: {', '.join(missing)}")

        with self._stage(res, "4. Проверка набора требований"):
            reqs_block = "\n".join(f"- {r['rid']} [{r['doc_id'] or '—'}] ({r['kind']}) L{r['lines'][0]}: {r['summary']}"
                                   for r in res.requirements)
            out = self.llm.json(prompts.SYSTEM, prompts.SET_LEVEL.format(
                criteria=prompts.set_criteria_block(), reqs=reqs_block, hints=hints, doc=numbered), stage="set")
            res.set_verdicts = {k: _verdict(out.get("verdicts", {}).get(k)) for k in SET_CRITERIA}
            res.coverage = out.get("coverage", [])
            for f in out.get("findings", []):
                f.update(scope="set", source="llm")
                raw.append(f)

        # правила тоже идут через рецензента и проверку цитат
        raw.extend(rule_findings)
        raw = self._aggregate_ids(res, raw)
        with self._stage(res, "5. Проверка цитат (анти-галлюцинации)"):
            checked = []
            for f in raw:
                if self._verify_evidence(doc, f):
                    self._attach_rids(res, f)
                    checked.append(f)
                else:
                    f["reject_reason"] = "цитата не найдена в документе"
                    res.rejected.append(f)
            for n, f in enumerate(_sort(checked), start=1):
                f["id"] = f"D-{n:02d}"
            self.log(f"   подтверждено цитатой: {len(checked)}, отброшено: {len(res.rejected)}")

        if self.use_critic and checked:
            with self._stage(res, "6. Независимый рецензент"):
                checked = self._critic(numbered, checked, res)
        res.findings = _sort(checked)
        for n, f in enumerate(res.findings, start=1):
            f["old_id"], f["id"] = f.get("id"), f"D-{n:02d}"
        # перенумерация ссылок в вердиктах
        self._sync_verdicts(res)

        with self._stage(res, "7. Уточняющие вопросы и итог"):
            out = self.llm.json(prompts.SYSTEM, prompts.SYNTHESIS.format(
                findings=_findings_block(res.findings), doc=numbered), stage="synthesis")
            res.questions = out.get("questions", [])
            res.verdict = out.get("verdict", "no") if out.get("verdict") in ("no", "conditional", "yes") else "no"
            res.verdict_reason = out.get("verdict_reason", "")
            res.priorities = out.get("priorities", [])
            # страховка: при критичных дефектах итог не может быть «yes»
            if any(f["severity"] == "critical" for f in res.findings) and res.verdict == "yes":
                res.verdict = "conditional"
        res.finished = time.time()
        return res

    # ------------------------------------------------------------------
    def clarify(self, res: RunResult, answers: dict[str, str]) -> None:
        """Учитывает ответы автора требований на уточняющие вопросы."""
        if not answers:
            return
        qa = "\n".join(f"{q['id']}. {q['question']}\nОтвет: {answers[q['id']]}"
                       for q in res.questions if answers.get(q["id"]))
        out = self.llm.json(prompts.SYSTEM, prompts.REASSESS.format(
            qa=qa, findings=_findings_block(res.findings)), stage="reassess")
        by_id = {f["id"]: f for f in res.findings}
        for u in out.get("updates", []):
            f = by_id.get(u.get("finding_id"))
            if f:
                f["clarification"] = {"status": u.get("status"), "note": u.get("note", "")}
        res.clarifications = {"answers": answers, "doc_changes": out.get("doc_changes", []),
                              "verdict": out.get("verdict"), "verdict_reason": out.get("verdict_reason", "")}

    # ------------------------------------------------------------------
    def _normalize_reqs(self, doc, items, res):
        out = []
        for it in items:
            quote = (it.get("quote") or "").strip()
            lines = _lines(it.get("lines"))
            status, a, b = doc.locate(quote, near=lines[0] if lines else None)
            if status == "missing":
                if lines and 1 <= lines[0] <= len(doc.lines):
                    quote = " ".join(doc.lines[i - 1].text.strip() for i in range(lines[0], min(lines[1], len(doc.lines)) + 1))[:220]
                    status, a, b = "line", lines[0], lines[1]
                else:
                    res.warnings.append(f"Пропущено требование без привязки к тексту: «{quote[:80]}»")
                    continue
            out.append({
                "rid": f"R-{len(out) + 1:02d}", "doc_id": it.get("doc_id") or None,
                "section": it.get("section") or "", "kind": it.get("kind") or "other",
                "summary": it.get("summary") or quote[:80], "quote": quote,
                "lines": [a, b], "atoms": it.get("atoms") or [], "evidence": status,
            })
        return out

    def _verify_evidence(self, doc, f) -> bool:
        lines = _lines(f.get("lines"))
        status, a, b = doc.locate(f.get("quote") or "", near=lines[0] if lines else None)
        if status == "missing":
            return False
        f["lines"], f["evidence"] = [a, b], status
        f["location"] = doc.location(a, b)
        if f.get("quote2"):
            l2 = _lines(f.get("lines2"))
            s2, a2, b2 = doc.locate(f["quote2"], near=l2[0] if l2 else None)
            if s2 == "missing":
                f["quote2"] = None
            else:
                f["lines2"], f["location2"] = [a2, b2], doc.location(a2, b2)
        f["severity"] = f.get("severity") if f.get("severity") in SEVERITY_ORDER else "major"
        f.setdefault("related_rids", [])
        return True

    @staticmethod
    def _aggregate_ids(res, raw):
        """Вместо N замечаний «нет идентификатора» — одно замечание о трассируемости набора."""
        id_pat = re.compile(r"идентификатор|нумерац|номер требован|\bID\b", re.I)
        noise = [f for f in raw if f.get("criterion") in ("conforming", "traceable")
                 and id_pat.search(f"{f.get('title', '')} {f.get('explanation', '')}")]
        rest = [f for f in raw if f not in noise]
        for f in noise:
            f["reject_reason"] = "объединено в замечание о трассируемости набора"
            res.rejected.append(f)
        unnum = [r for r in res.requirements if not r["doc_id"]]
        if unnum:
            ids = ", ".join(r["rid"] for r in unnum)
            rest.append({
                "source": "rule", "scope": "set", "criterion": "traceable", "severity": "minor",
                "title": f"{len(unnum)} из {len(res.requirements)} требований без идентификаторов",
                "quote": unnum[0]["quote"], "lines": unnum[0]["lines"],
                "explanation": f"Без идентификатора на требование нельзя сослаться из тест-кейса, задачи "
                               f"или замечания ревью. Без номеров: {ids}.",
                "recommendation": "Присвоить сквозные идентификаторы (ФТ-, НФТ-, ДАН-, ИНТ-, API-, AC-) и "
                                  "добавить матрицу трассируемости «цель — требование — критерий приёмки».",
                "related_rids": [],
            })
        return rest

    @staticmethod
    def _attach_rids(res, f):
        valid = {r["rid"] for r in res.requirements}
        f["related_rids"] = [x for x in (f.get("related_rids") or []) if x in valid]
        if f["related_rids"] or f.get("scope") == "set" and f.get("source") == "llm":
            return
        a, b = f["lines"]
        hits = [r["rid"] for r in res.requirements if r["lines"][0] <= b and a <= r["lines"][1]]
        f["related_rids"] = hits[:3]

    def _critic(self, numbered, findings, res):
        try:
            out = self.llm.json(prompts.SYSTEM, prompts.CRITIC.format(
                findings=_findings_block(findings), doc=numbered), stage="critic")
        except LLMError as e:
            res.warnings.append(f"Рецензент недоступен, замечания не фильтровались: {e}")
            return findings
        dec = {d.get("id"): d for d in out.get("decisions", [])}
        by_id = {f["id"]: f for f in findings}
        kept = []
        for f in findings:
            d = dec.get(f["id"])
            if not d:
                kept.append(f)
                continue
            act = d.get("action")
            if act == "drop" and f.get("severity") == "critical":
                # критичный дефект с подтверждённой цитатой не удаляем молча: понижаем и помечаем
                f["severity"] = "major"
                f["critic"] = f"рецензент сомневается: {d.get('reason', '')}"
                kept.append(f)
            elif act == "drop":
                f["reject_reason"] = f"рецензент: {d.get('reason', '')}"
                res.rejected.append(f)
            elif act == "merge" and d.get("into") in by_id and d.get("into") != f["id"]:
                tgt = by_id[d["into"]]
                tgt.setdefault("merged", []).append(f["id"])
                tgt.setdefault("also", []).append({"title": f.get("title"), "quote": f.get("quote"),
                                                   "location": f.get("location"), "explanation": f.get("explanation")})
                tgt["related_rids"] = sorted(set(tgt.get("related_rids", []) + f.get("related_rids", [])))
                if SEVERITY_ORDER.get(f["severity"], 9) < SEVERITY_ORDER.get(tgt["severity"], 9):
                    tgt["severity"] = f["severity"]
            else:
                if d.get("severity") in SEVERITY_ORDER:
                    f["severity"] = d["severity"]
                f["critic"] = d.get("reason", "")
                kept.append(f)
        # если merge указывал на удалённое — вернём
        kept_ids = {f["id"] for f in kept}
        for f in findings:
            d = dec.get(f["id"]) or {}
            if d.get("action") == "merge" and d.get("into") not in kept_ids and f not in kept:
                kept.append(f)
        self.log(f"   оставлено: {len(kept)}, отсеяно рецензентом: {len(findings) - len(kept)}")
        return kept

    def _sync_verdicts(self, res):
        """Если по требованию не осталось замечаний по критерию — критерий не может быть 'issue'."""
        for r in res.requirements:
            v = res.req_verdicts.get(r["rid"])
            if not v:
                continue
            crits = {f["criterion"] for f in res.findings if r["rid"] in f.get("related_rids", [])}
            for k, val in v.items():
                if val == "issue" and k not in crits:
                    v[k] = "ok"
            for k in crits:
                if k in v:
                    v[k] = "issue"

    class _Timer:
        def __init__(self, agent, res, name):
            self.a, self.res, self.name = agent, res, name

        def __enter__(self):
            self.t = time.time()
            self.a.log(f"▶ {self.name}")

        def __exit__(self, *exc):
            self.res.stage_times[self.name] = round(time.time() - self.t, 1)
            return False

    def _stage(self, res, name):
        return Agent._Timer(self, res, name)


def _verdict(v):
    return v if v in ("ok", "issue", "na") else "na"


def _lines(v):
    try:
        a, b = int(v[0]), int(v[1] if len(v) > 1 else v[0])
        return [a, max(a, b)]
    except (TypeError, ValueError, IndexError):
        return None


def _sort(findings):
    return sorted(findings, key=lambda f: (SEVERITY_ORDER.get(f.get("severity"), 9),
                                           0 if f.get("scope") == "set" else 1,
                                           (f.get("lines") or [10 ** 6])[0]))


def _findings_block(findings):
    rows = []
    for f in findings:
        s = (f"{f['id']} [{f['severity']}] [{f['criterion']}] {f.get('location', '')} "
             f"({', '.join(f.get('related_rids') or []) or 'набор'}): {f['title']}. "
             f"Цитата: «{f['quote'][:160]}»")
        if f.get("quote2"):
            s += f" ↔ «{f['quote2'][:120]}»"
        s += f". {f.get('explanation', '')[:300]}"
        rows.append(s)
    return "\n".join(rows)


def to_json(res: RunResult) -> dict:
    return {
        "document": {"path": str(res.doc.path), "title": res.doc.title, "format": res.doc.fmt,
                     "sha256": res.doc.sha256, "lines": len(res.doc.lines)},
        "model": res.model, "started": res.started, "finished": res.finished,
        "requirements": res.requirements, "req_verdicts": res.req_verdicts, "rewrites": res.rewrites,
        "set_verdicts": res.set_verdicts, "coverage": res.coverage, "findings": res.findings,
        "rejected": res.rejected, "questions": res.questions, "verdict": res.verdict,
        "verdict_reason": res.verdict_reason, "priorities": res.priorities,
        "clarifications": res.clarifications, "warnings": res.warnings, "stage_times": res.stage_times,
    }


def dumps(res: RunResult) -> str:
    return json.dumps(to_json(res), ensure_ascii=False, indent=2, default=str)
