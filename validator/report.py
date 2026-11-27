"""Формирование отчёта в Markdown (на русском)."""
from __future__ import annotations

import re
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import __version__
from .criteria import REQ_CRITERIA, SET_CRITERIA, SEVERITY_RU, criterion_label
from .pipeline import RunResult

ICON = {"ok": "✅", "issue": "❌", "na": "—"}
SEV_ICON = {"critical": "🔴", "major": "🟠", "minor": "🟡"}
VERDICT_RU = {
    "no": "❌ Начинать разработку и тестирование по документу **нельзя**",
    "conditional": "⚠️ Начинать можно **только после исправления** критичных дефектов",
    "yes": "✅ Документ готов к разработке и тестированию",
}
KIND_RU = {"goal": "цель", "functional": "ФТ", "nfr": "НФТ", "data": "данные", "integration": "интеграция",
           "ui": "UI", "api": "API", "acceptance": "приёмка", "process": "процесс", "other": "прочее"}

UNVERIFIABLE = {"verifiable"}
CONTRADICTORY = {"correct", "set_consistent"}
INCOMPLETE = {"complete", "set_complete"}


def agent_version() -> str:
    """Версия агента и коммит. Работает и без установленного git: читает .git напрямую."""
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "-C", str(root), "describe", "--tags", "--always", "--dirty"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        commit = ""
    if not commit:
        commit = _read_git(root / ".git")
    return f"{__version__} ({commit})" if commit else __version__


def _read_git(g: Path) -> str:
    try:
        head = (g / "HEAD").read_text().strip()
        sha = head
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            f = g / ref
            if f.exists():
                sha = f.read_text().strip()
            else:
                packed = (g / "packed-refs").read_text() if (g / "packed-refs").exists() else ""
                sha = next((l.split()[0] for l in packed.splitlines() if l.endswith(" " + ref)), "")
        if not sha:
            return ""
        tags = []
        tdir = g / "refs" / "tags"
        if tdir.exists():
            tags = [t.name for t in tdir.iterdir() if t.read_text().strip() == sha]
        if (g / "packed-refs").exists():
            tags += [l.split()[1].split("/")[-1] for l in (g / "packed-refs").read_text().splitlines()
                     if l.startswith(sha) and "refs/tags/" in l]
        return f"{sorted(tags)[-1]}, коммит {sha[:7]}" if tags else f"коммит {sha[:7]}"
    except Exception:
        return ""


_LREF2 = re.compile(r"\bL(\d+)\s*[-–—]\s*L?(\d+)\b")
_LREF = re.compile(r"\bL(\d+)\b")


def esc(s) -> str:
    s = str(s or "").replace("|", "\\|").replace("\n", " ").strip()
    s = _LREF2.sub(r"строки \1–\2", s)
    return _LREF.sub(r"строка \1", s)


def render(res: RunResult, llm=None) -> str:
    llm_stats = getattr(llm, "stats", None)
    d = res.doc
    tz = timezone(timedelta(hours=5))
    started = datetime.fromtimestamp(res.started, tz)
    F = res.findings
    by_rid: dict[str, list] = {}
    for f in F:
        for rid in f.get("related_rids") or []:
            by_rid.setdefault(rid, []).append(f)
    sev = Counter(f["severity"] for f in F)
    out: list[str] = []
    w = out.append

    w(f"# Отчёт о проверке требований: «{esc(d.title)}»\n")
    w("> Отчёт сформирован автоматически агентом проверки требований по ISO/IEC/IEEE 29148:2018 "
      "(п. 5.2.5, 5.2.6) и IEEE Std 830-1998 (п. 4.3). Каждое замечание привязано к строке документа "
      "и дословной цитате; цитаты проверены программно.\n")

    # ---------------- паспорт ----------------
    w("## 0. Паспорт прогона\n")
    w("| Поле | Значение |\n| :--- | :--- |")
    w(f"| Документ | {esc(d.title)} |")
    w(f"| Файл | `{d.path.name}` ({d.fmt}, {len(d.lines)} строк) |")
    w(f"| SHA-256 файла | `{d.sha256[:16]}…` |")
    w(f"| Дата и время прогона | {started:%d.%m.%Y %H:%M} (UTC+5, Алматы) |")
    temp = getattr(llm, "temperature", None)
    w(f"| Модель | `{res.model}`" + (f" (temperature = {temp:g})" if temp is not None else "") + " |")
    w(f"| Версия агента | `{agent_version()}` |")
    dur = res.finished - res.started
    replay = bool(llm_stats) and llm_stats.calls == 0 and llm_stats.cached > 0
    if replay:
        w(f"| Режим | **Воспроизведение из кэша**: все {llm_stats.cached} ответов модели взяты из `.cache/` "
          f"исходного прогона; новых запросов к API — 0. Тот же документ, код и кэш дают тот же отчёт. |")
    elif llm_stats:
        tin = f"{llm_stats.input_tokens:,}".replace(",", " ")
        tout = f"{llm_stats.output_tokens:,}".replace(",", " ")
        w(f"| Обращений к модели | {llm_stats.calls} (+{llm_stats.cached} из кэша), токены: вход {tin}, выход {tout} |")
    w(f"| Длительность | {dur:.0f} с" + (" (без обращений к модели)" if replay else "") + " |")
    if not replay:
        w(f"| Этапы | " + "; ".join(f"{k} — {v} с" for k, v in res.stage_times.items()) + " |")
    w("")

    # ---------------- кратко ----------------
    w("### Кратко\n")
    w(f"- {VERDICT_RU.get(res.verdict, res.verdict)}.")
    w(f"- Требований выделено: **{len(res.requirements)}**; дефектов: **{len(F)}** "
      f"({SEV_ICON['critical']} критичных — {sev['critical']}, {SEV_ICON['major']} существенных — {sev['major']}, "
      f"{SEV_ICON['minor']} незначительных — {sev['minor']}).")
    w(f"- Уточняющих вопросов автору: **{len(res.questions)}** (раздел 5).\n")

    # ---------------- 1. нарезка ----------------
    w("## 1. Как агент нарезал требования\n")
    w("Номера документа сохранены там, где они есть; где номеров нет, агент присвоил свои (`R-xx`) "
      "и сохранил цитату.\n")
    w("| ID в отчёте | ID в документе | Тип | Раздел | Где | Краткая формулировка | Цитата |")
    w("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in res.requirements:
        w(f"| **{r['rid']}** | {esc(r['doc_id']) or '_нет, присвоен агентом_'} | {KIND_RU.get(r['kind'], r['kind'])} | "
          f"{esc(r['section'])} | {d.location(*r['lines'])} | {esc(r['summary'])} | «{esc(r['quote'][:140])}» |")
    w("")

    # ---------------- 2. каждое требование ----------------
    w("## 2. Проверка каждого требования\n")
    w("### 2.1. Матрица соответствия критериям ISO/IEC/IEEE 29148 §5.2.5\n")
    keys = list(REQ_CRITERIA)
    w("| ID | " + " | ".join(REQ_CRITERIA[k][0] for k in keys) + " | Дефекты |")
    w("| :--- |" + " :---: |" * len(keys) + " :--- |")
    for r in res.requirements:
        v = res.req_verdicts.get(r["rid"], {})
        ids = ", ".join(f["id"] for f in by_rid.get(r["rid"], []))
        w(f"| {r['rid']} | " + " | ".join(ICON.get(v.get(k, "na"), "—") for k in keys) + f" | {ids or '—'} |")
    w("\n✅ — соответствует, ❌ — нарушение (см. дефект), — — неприменимо/не оценивалось.\n")

    w("### 2.2. Замечания по требованиям\n")
    for r in res.requirements:
        fl = [f for f in by_rid.get(r["rid"], [])]
        head = f"#### {r['rid']}" + (f" ({esc(r['doc_id'])})" if r["doc_id"] else "") + f" — {esc(r['summary'])}"
        w(head + "\n")
        w(f"> {esc(r['quote'])}  \n> — _{d.location(*r['lines'])}, раздел «{esc(r['section'])}»_\n")
        if not fl:
            w("Нарушений не выявлено.\n")
        for f in fl:
            if f.get("scope") == "set":
                w(f"- {SEV_ICON[f['severity']]} см. {f['id']} в разделе 3 (проверка набора): {esc(f['title'])}\n")
                continue
            if r["rid"] != (f.get("related_rids") or [None])[0]:
                w(f"- {SEV_ICON[f['severity']]} см. {f['id']} в {f['related_rids'][0]}: {esc(f['title'])}\n")
                continue
            _finding(w, f)
        if r["rid"] in res.rewrites:
            w(f"**Предлагаемая формулировка:** {esc(res.rewrites[r['rid']])}\n")

    # ---------------- 3. набор ----------------
    w("## 3. Проверка всего набора требований\n")
    w("### 3.1. Свойства набора (ISO/IEC/IEEE 29148 §5.2.6, IEEE 830 §4.3)\n")
    w("| Свойство | Источник | Оценка | Дефекты |\n| :--- | :--- | :---: | :--- |")
    for k, (name, ref, _) in SET_CRITERIA.items():
        ids = ", ".join(f["id"] for f in F if f["criterion"] == k)
        model_v = res.set_verdicts.get(k, "na")
        mark = "❌" if ids else ("⚠️" if model_v == "issue" else ICON.get(model_v, "—"))
        w(f"| {name} | {ref} | {mark} | {ids or '—'} |")
    w("\n❌ — нарушение (см. дефекты); ⚠️ — модель сомневается, отдельного подтверждённого дефекта нет; ✅ — соответствует.\n")
    w("### 3.2. Замечания уровня набора (противоречия, пропуски, согласованность)\n")
    set_f = [f for f in F if f.get("scope") == "set"]
    if not set_f:
        w("Нарушений не выявлено.\n")
    for f in set_f:
        _finding(w, f)
    if res.coverage:
        rid2req = {r["rid"]: r for r in res.requirements}
        uncovered = [c["rid"] for c in res.coverage if not c.get("covered_by_acceptance") and c.get("rid") in rid2req
                     and rid2req[c["rid"]]["kind"] not in ("acceptance", "goal")]
        if uncovered:
            w("### 3.3. Покрытие критериями приёмки\n")
            w(f"Не покрыты ни одним критерием приёмки: **{len(uncovered)}** из "
              f"{sum(1 for r in res.requirements if r['kind'] not in ('acceptance', 'goal'))} требований — "
              + ", ".join(f"{x}" + (f" ({rid2req[x]['doc_id']})" if rid2req[x]['doc_id'] else '') for x in uncovered) + ".\n")

    # ---------------- 4. сводка ----------------
    w("## 4. Сводка\n")
    w("| Показатель | Значение |\n| :--- | :--- |")
    w(f"| Требований выделено | {len(res.requirements)} |")
    w(f"| Требований с нарушениями | {sum(1 for r in res.requirements if by_rid.get(r['rid']))} |")
    w(f"| Всего дефектов | {len(F)} |")
    for s in ("critical", "major", "minor"):
        w(f"| {SEV_ICON[s]} {SEVERITY_RU[s]} | {sev[s]} |")
    w(f"| Отсеяно как неподтверждённые | {len(res.rejected)} (см. приложение) |\n")
    crit_count = Counter(f["criterion"] for f in F)
    w("**Нарушения по критериям:** " + "; ".join(f"{criterion_label(k)} — {v}" for k, v in crit_count.most_common()) + ".\n")

    w("### Критичные дефекты по группам\n")
    w("Критичные (🔴) и существенные (🟠) дефекты, сгруппированные так, как требует ТЗ.\n")
    groups = [("Непроверяемые формулировки", UNVERIFIABLE), ("Противоречивые формулировки", CONTRADICTORY),
              ("Неполные формулировки", INCOMPLETE)]
    shown = set()
    for title, crits in groups:
        lst = [f for f in F if f["criterion"] in crits and f["severity"] in ("critical", "major")]
        if lst:
            w(f"**{title}:**\n")
            for f in lst:
                w(f"- {SEV_ICON[f['severity']]} **{f['id']}** — {esc(f['title'])} ({f.get('location', '')})")
                shown.add(f["id"])
            w("")
    other = [f for f in F if f["severity"] == "critical" and f["id"] not in shown]
    if other:
        w("**Прочие критичные:**\n")
        for f in other:
            w(f"- {SEV_ICON['critical']} **{f['id']}** — {esc(f['title'])} ({f.get('location', '')})")
        w("")

    # ---------------- 5. вопросы ----------------
    w("## 5. Уточняющие вопросы к автору требований\n")
    w("Без ответов на эти вопросы разработчик и тестировщик будут вынуждены угадывать поведение системы.\n")
    for q in res.questions:
        w(f"**{q.get('id')}.** _{esc(q.get('to', ''))}_ — {esc(q.get('question'))}  ")
        if q.get("why"):
            w(f"Зачем: {esc(q['why'])}  ")
        if q.get("options"):
            w("Варианты: " + " / ".join(esc(o) for o in q["options"]) + "  ")
        if q.get("default"):
            w(f"Предложение агента: {esc(q['default'])}  ")
        if q.get("finding_ids"):
            w(f"Связанные дефекты: {', '.join(q['finding_ids'])}")
        if res.clarifications.get("answers", {}).get(q.get("id")):
            w(f"\n> **Ответ автора:** {esc(res.clarifications['answers'][q['id']])}")
        w("")
    if res.clarifications:
        w("### 5.1. Учёт полученных ответов\n")
        resolved = [f for f in F if (f.get("clarification") or {}).get("status") == "resolved"]
        w(f"Ответы закрывают дефекты: {', '.join(f['id'] for f in resolved) or 'нет'}.\n")
        for ch in res.clarifications.get("doc_changes", []):
            w(f"- Внести в документ: {esc(ch)}")
        w("")

    # ---------------- 6. итог ----------------
    w("## 6. Итог\n")
    w(f"### {VERDICT_RU.get(res.verdict, res.verdict)}\n")
    w(esc(res.verdict_reason) + "\n")
    if res.clarifications.get("verdict"):
        w(f"С учётом ответов автора: **{VERDICT_RU.get(res.clarifications['verdict'], '')}**. "
          f"{esc(res.clarifications.get('verdict_reason'))}\n")
    w("### Что исправить в первую очередь\n")
    for i, p in enumerate(res.priorities, 1):
        ids = ", ".join(p.get("finding_ids") or [])
        w(f"{i}. {esc(p.get('action'))}" + (f" ({ids})" if ids else ""))
    w("")

    # ---------------- приложение ----------------
    w("## Приложение А. Методика\n")
    w("1. Детерминированный линтер: размытые слова, заглушки, незаполненные поля и чек-листы, единицы измерения, "
      "HTTP-методы и коды ответов.\n"
      "2. Модель нарезает документ на требования с цитатами и номерами строк.\n"
      "3. Модель оценивает каждое требование по 9 свойствам ISO/IEC/IEEE 29148 §5.2.5 с учётом всего документа.\n"
      "4. Модель проверяет набор целиком по §5.2.6 и IEEE 830 §4.3: противоречия, полнота, покрытие приёмкой.\n"
      "5. Каждая цитата ищется в исходном тексте программно; замечания с ненайденной цитатой отбрасываются.\n"
      "6. Независимый проход-рецензент удаляет ложные срабатывания и дубликаты, калибрует серьёзность.\n"
      "7. По подтверждённым дефектам формируются уточняющие вопросы, итог и приоритеты.\n")
    w("Серьёзность: 🔴 критичный — разработка/тест пойдут по неверному пути или результат нельзя принять; "
      "🟠 существенный — почти наверняка разное толкование; 🟡 незначительный — оформление.\n")
    if res.rejected:
        w("## Приложение Б. Отсеянные замечания\n")
        w("Показаны для прозрачности: в подсчёты не входят.\n")
        w("| Замечание | Причина |\n| :--- | :--- |")
        for f in res.rejected:
            w(f"| {esc(f.get('title'))} — «{esc((f.get('quote') or '')[:80])}» | {esc(f.get('reject_reason'))} |")
        w("")
    if res.warnings:
        w("## Приложение В. Предупреждения прогона\n")
        for x in res.warnings:
            w(f"- {esc(x)}")
    return "\n".join(out) + "\n"


def _finding(w, f):
    src = "правило" if f.get("source") == "rule" else "модель"
    w(f"- {SEV_ICON[f['severity']]} **{f['id']}. {esc(f['title'])}** — {SEVERITY_RU[f['severity']].lower()}, "
      f"_{criterion_label(f['criterion'])}_, {f.get('location', '')}")
    w(f"  - Цитата: «{esc(f['quote'])}»")
    if f.get("quote2"):
        w(f"  - Противоречит: «{esc(f['quote2'])}» ({f.get('location2', '')})")
    if f.get("explanation"):
        w(f"  - Проблема: {esc(f['explanation'])}")
    if f.get("recommendation"):
        w(f"  - Как исправить: {esc(f['recommendation'])}")
    for a in f.get("also") or []:
        w(f"  - Тот же дефект проявляется: «{esc(a.get('quote'))}» ({a.get('location', '')}) — {esc(a.get('title'))}")
    extra = []
    if f.get("evidence") == "fuzzy":
        extra.append("цитата найдена с небольшими расхождениями")
    extra.append(f"источник: {src}")
    w(f"  - _{'; '.join(extra)}_")
    if f.get("clarification"):
        c = f["clarification"]
        w(f"  - После уточнения: **{'закрыт' if c.get('status') == 'resolved' else 'остаётся открытым'}** — {esc(c.get('note'))}")
    w("")
