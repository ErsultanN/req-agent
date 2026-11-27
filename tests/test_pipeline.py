"""Сквозной тест конвейера без сети: модель подменена сценарием (FakeLLM).
Запуск: python -m pytest tests -q   (или python tests/test_pipeline.py)"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from validator import document, report, rules  # noqa: E402
from validator.llm import LLMStats, parse_json  # noqa: E402
from validator.pipeline import Agent, dumps  # noqa: E402

SAMPLE = ROOT / "samples" / "requirements.md"


class FakeLLM:
    label = "fake:scripted"

    def __init__(self):
        self.stats = LLMStats()
        self.stages = []

    def json(self, system, user, stage=""):
        self.stages.append(stage)
        self.stats.calls += 1
        if stage == "segment":
            return {"document_title": "Уведомление о расходах по карте", "requirements": [
                {"doc_id": "Ф-1", "section": "Функциональные требования", "kind": "functional",
                 "summary": "Порог 1 000–1 000 000 ₸", "quote": "задаёт целый порог от 1 000 до 1 000 000 ₸ включительно",
                 "lines": [30, 30], "atoms": ["порог", "по умолчанию выключено"]},
                {"doc_id": "Ф-3", "section": "Функциональные требования", "kind": "functional",
                 "summary": "Push при достижении порога", "quote": "На карту приходится максимум один такой push за календарный месяц",
                 "lines": [32, 32]},
                {"doc_id": "Ф-4", "section": "Функциональные требования", "kind": "functional",
                 "summary": "Смена порога", "quote": "следующее достижение вызывает push независимо от истории предыдущей настройки",
                 "lines": [33, 33]},
                {"doc_id": None, "section": "Выдуманный", "kind": "other", "summary": "галлюцинация",
                 "quote": "Система поддерживает биометрию", "lines": None},
            ]}
        if stage == "per_req":
            return {"results": [
                {"rid": "R-01", "verdicts": {"singular": "issue"}, "findings": [
                    {"criterion": "singular", "severity": "minor", "title": "Два условия в одном пункте",
                     "quote": "По умолчанию уведомление выключено.", "lines": [30, 30],
                     "explanation": "…", "recommendation": "Разделить"}]},
                {"rid": "R-02", "verdicts": {"correct": "issue"}, "findings": [
                    {"criterion": "correct", "severity": "critical", "title": "Противоречие Ф-3 и Ф-4",
                     "quote": "максимум один такой push за календарный месяц", "lines": [32, 32],
                     "explanation": "…", "recommendation": "…", "related_rids": ["R-03"]},
                    {"criterion": "verifiable", "severity": "major", "title": "Галлюцинация",
                     "quote": "push отправляется в течение 5 секунд", "lines": [32, 32],
                     "explanation": "", "recommendation": ""}]},
            ]}
        if stage == "set":
            return {"verdicts": {"set_consistent": "issue"}, "findings": [
                {"criterion": "set_consistent", "severity": "critical", "title": "Приёмка против Ф-2",
                 "quote": "Февральский счётчик вырос на 4 000 ₸", "lines": [64, 64],
                 "quote2": "Месяц определяется по дате совершения покупки в Алматы", "lines2": [31, 31],
                 "explanation": "…", "recommendation": "…", "related_rids": ["R-01"]}],
                "coverage": [{"rid": "R-01", "covered_by_acceptance": False}]}
        if stage == "critic":
            ids = re.findall(r"^(D-\d+)", user, re.M)
            return {"decisions": [{"id": i, "action": "keep", "severity": None, "reason": "ок"} for i in ids]}
        if stage == "synthesis":
            return {"questions": [{"id": "Q1", "to": "бизнес", "question": "К какому месяцу относить покупку?",
                                   "why": "…", "options": ["по дате покупки", "по дате проведения"],
                                   "default": "по дате покупки", "finding_ids": ["D-01"]}],
                    "verdict": "no", "verdict_reason": "Есть критичные противоречия.",
                    "priorities": [{"action": "Согласовать Ф-2 и приёмку", "finding_ids": ["D-01"]}]}
        if stage == "reassess":
            return {"updates": [{"finding_id": "D-01", "status": "resolved", "note": "ок"}],
                    "doc_changes": ["Исправить критерий приёмки №2"], "verdict": "conditional", "verdict_reason": "…"}
        raise AssertionError(stage)


def test_locate():
    d = document.load(SAMPLE)
    assert d.locate("Февральский счетчик вырос на 4000 ₸")[0] in ("exact", "fuzzy")
    assert d.locate("Система поддерживает биометрию")[0] == "missing"


def test_rules():
    d = document.load(SAMPLE)
    titles = " ".join(f["title"] for f in rules.run(d))
    assert "GET" in titles and "тиыны" in titles and "201" in titles


def test_pipeline_end_to_end(tmp_path=Path("/tmp")):
    d = document.load(SAMPLE)
    llm = FakeLLM()
    agent = Agent(llm, log=lambda *a: None)
    res = agent.run(d)
    assert len(res.requirements) == 3, "галлюцинированное требование должно быть отброшено"
    assert any("Галлюцинация" in f["title"] for f in res.rejected)
    assert res.findings[0]["severity"] == "critical"
    assert all(f.get("location") for f in res.findings)
    agent.clarify(res, {"Q1": "по дате покупки"})
    md = report.render(res, llm)
    for section in ("Паспорт прогона", "Как агент нарезал", "Проверка каждого", "Проверка всего набора",
                    "Сводка", "Итог", "Уточняющие вопросы"):
        assert section in md, section
    json.loads(dumps(res))
    (tmp_path / "fake_report.md").write_text(md, "utf-8")


def test_parse_json():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('текст {"a": [1]} хвост') == {"a": [1]}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
