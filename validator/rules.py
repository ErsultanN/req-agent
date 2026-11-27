"""Детерминированные проверки (линтер требований).

Работают без LLM, дают воспроизводимую основу и подсказки модели. Правила общие —
не привязаны к конкретному документу.
"""
from __future__ import annotations

import re

from .document import Document

# Размытые слова и обороты (критерий unambiguous / verifiable)
VAGUE = [
    r"быстр\w*", r"удобн\w*", r"оперативн\w*", r"своевременн\w*", r"при необходимости", r"по возможности",
    r"и т\.\s?д\.?", r"и т\.\s?п\.?", r"и др\.", r"достаточн\w*", r"максимальн\w*", r"оптимальн\w*",
    r"интуитивн\w*", r"гибк\w*", r"примерно", r"около", r"как правило", r"обычно", r"надёжн\w*", r"надежн\w*",
    r"эффективн\w*", r"современн\w*", r"понятн\w*", r"минимальн\w*", r"в момент", r"мгновенн\w*",
    r"user[- ]friendly", r"и/или", r"в разумные сроки", r"при желании", r"нормальн\w*",
]
PLACEHOLDERS = [r"\bTBD\b", r"\bTODO\b", r"\bуточн(ить|яется)\b", r"заполня\w+ (позже|тестировщик\w*|аналитик\w*)",
                r"\?\?\?", r"\bXXX\b", r"в источнике не заполнен", r"не заполнен\w*", r"отметки не проставлены"]
UNFILLED_CHOICE = re.compile(r"Да\s*/\s*Нет")
SAVE_VERBS = re.compile(r"(save|create|update|delete|remove|set|add|сохран|созда|измен|удал)", re.I)
HTTP_CODE = re.compile(r"(?<![\d.,])([1-5]\d\d)(?![\d.,])")


def _f(doc: Document, line_no: int, criterion: str, severity: str, title: str, explanation: str,
       recommendation: str, scope: str = "requirement", quote: str | None = None):
    text = doc.lines[line_no - 1].text.strip()
    return {
        "source": "rule", "scope": scope, "criterion": criterion, "severity": severity, "title": title,
        "quote": (quote or text)[:220], "lines": [line_no, line_no],
        "explanation": explanation, "recommendation": recommendation, "related_rids": [],
    }


def run(doc: Document) -> list[dict]:
    out: list[dict] = []
    lines = doc.lines
    txt_all = "\n".join(l.text for l in lines)

    for l in lines:
        t = l.text
        if not t.strip():
            continue
        is_table_sep = re.fullmatch(r"[\s|:\-]+", t) is not None and "-" in t
        if is_table_sep:
            continue
        # 1) размытые формулировки
        for pat in VAGUE:
            m = re.search(pat, t, re.I)
            if m:
                out.append(_f(doc, l.no, "unambiguous", "minor", f"Размытая формулировка «{m.group(0)}»",
                              f"Слово «{m.group(0)}» допускает разные толкования и не даёт измеримого критерия.",
                              "Заменить измеримым условием (число, единица, перцентиль, точное событие).",
                              quote=_ctx(t, m)))
                break
        # 2) заглушки и незаполненные места
        for pat in PLACEHOLDERS:
            m = re.search(pat, t, re.I)
            if m:
                out.append(_f(doc, l.no, "set_complete", "major", "Незаполненный раздел или заглушка",
                              f"Текст «{m.group(0)}» означает, что содержание раздела отсутствует.",
                              "Заполнить раздел до передачи в разработку.", scope="set", quote=_ctx(t, m)))
                break
        # 3) таблица с пустыми значениями / прочерками
        if t.strip().startswith("|"):
            cells = [c.strip() for c in t.strip().strip("|").split("|")]
            if cells and all(c == "" for c in cells):
                out.append(_f(doc, l.no, "set_complete", "minor", "Пустая строка таблицы",
                              "Строка таблицы не заполнена (например, блок согласующих).",
                              "Заполнить или удалить строку.", scope="set",
                              quote=_prev_nonempty(doc, l.no)))
            elif len(cells) == 2 and re.fullmatch(r"[—–-]", cells[1]):
                out.append(_f(doc, l.no, "complete", "major", "Значение поля не заполнено («—»)",
                              "В таблице стоит прочерк вместо значения.", "Указать значение или «не применимо» с причиной."))
        if re.search(r":\s*[—–]\s*(\.|\||$)", t):
            out.append(_f(doc, l.no, "complete", "major", "Значение не заполнено («—»)",
                          "После двоеточия стоит прочерк вместо значения.", "Указать значение."))

    # 4) незаполненный чек-лист — одно агрегированное замечание
    unfilled = [l for l in lines if UNFILLED_CHOICE.search(l.text)]
    if len(unfilled) >= 3:
        out.append(_f(doc, unfilled[0].no, "modifiable", "minor",
                      f"Чек-лист ревью не заполнен ({len(unfilled)} пунктов «Да / Нет»)",
                      "Варианты «Да / Нет» не выбраны: ревью требований формально не проведено.",
                      "Пройти чек-лист и проставить отметки с примечаниями.", scope="set"))
        out[-1]["lines"] = [unfilled[0].no, unfilled[-1].no]

    # 5) версия документа
    if not re.search(r"верси[яи]\s*[:№]?\s*\d", txt_all, re.I):
        ln = next((l.no for l in lines if re.search(r"верси", l.text, re.I)), 1)
        out.append(_f(doc, ln, "modifiable", "minor", "Не указаны версия и дата документа",
                      "Без версии нельзя сослаться на согласованную редакцию и отследить изменения.",
                      "Добавить версию, дату, историю изменений.", scope="set"))

    # 6) единицы денег: тиыны и тенге в одном документе
    tiyn = next((l for l in lines if re.search(r"тиын", l.text, re.I)), None)
    if tiyn and re.search(r"тенге|₸|KZT", txt_all):
        out.append(_f(doc, tiyn.no, "set_consistent", "major", "Смешаны единицы: тиыны и тенге",
                      "Суммы хранятся/передаются в разных единицах; правило пересчёта и сравнения с порогом не задано.",
                      "Зафиксировать одну единицу хранения и правило конвертации/округления.", scope="set"))

    # 7) REST: изменение данных через GET
    for l in lines:
        if re.search(r"\bGET\b", l.text):
            window = " ".join(x.text for x in lines[max(0, l.no - 8):l.no + 12])
            m = SAVE_VERBS.search(window)
            if m:
                out.append(_f(doc, l.no, "appropriate", "critical", "Изменение состояния через HTTP GET",
                              "Метод сохраняет данные, но объявлен как GET: GET должен быть безопасным и идемпотентным "
                              "(RFC 9110), параметры попадают в URL, логи и кэш прокси.",
                              "Использовать POST/PUT с телом запроса; идентификатор карты не передавать в query.",
                              scope="requirement"))
                break

    # 8) Тело запроса: Content-Type задан, но тело не используется
    ct = next((l for l in lines if re.search(r"Content-Type", l.text, re.I)), None)
    nobody = next((l for l in lines if re.search(r"тело не используется", l.text, re.I)), None)
    if ct and nobody and abs(ct.no - nobody.no) <= 4:
        out.append(_f(doc, ct.no, "set_consistent", "major", "Content-Type задан, а тело запроса не используется",
                      "Указан формат тела запроса, хотя сказано, что тела нет.", "Убрать Content-Type или описать тело.",
                      scope="set"))
        out[-1]["lines"] = [ct.no, nobody.no]

    # 9) HTTP-коды: заявленные vs описанные
    out.extend(_http_codes(doc))

    # 11) приоритеты требований
    if not re.search(r"приоритет|priority|MoSCoW|\bmust\b|\bshould\b|критичност", txt_all, re.I):
        ln = next((l.no for l in lines if re.search(r"функциональн\w* требован", l.text, re.I)), None)
        if ln:
            out.append(_f(doc, ln, "ranked", "minor", "У требований нет приоритетов",
                          "Ни одному требованию не присвоены важность и стабильность: при нехватке времени "
                          "неясно, что реализовать и тестировать в первую очередь.",
                          "Указать приоритет (например, MoSCoW) для каждого требования.", scope="set"))

    from .topics import missing_topics
    out.extend(missing_topics(doc, _f))
    return _dedupe(out)


def _http_codes(doc: Document) -> list[dict]:
    res = []
    mentioned: dict[str, int] = {}
    described: set[str] = set()
    for l in doc.lines:
        t = l.text.strip()
        if not re.search(r"HTTP|статус|ответ|результат|возвращ|код|result|\*\*\d{3}|#+\s*\d{3}|^\*\*\d{3}", t, re.I) and not re.match(r"^[#*\s]*\d{3}\b", t):
            continue
        for c in HTTP_CODE.findall(t):
            mentioned.setdefault(c, l.no)
        m = re.match(r"^[#*\s]*([1-5]\d\d)\b", t)
        if m:
            described.add(m.group(1))
    for code, ln in mentioned.items():
        if code not in described and code[0] in "2345":
            res.append(_f(doc, ln, "set_consistent", "major", f"HTTP {code} упомянут, но не описан",
                          f"Код {code} фигурирует в тексте, но для него нет раздела с условием и телом ответа.",
                          f"Описать условие и формат ответа {code} или убрать упоминание.", scope="set"))
    return res


def _ctx(t: str, m: re.Match, width: int = 90) -> str:
    a = max(0, m.start() - width // 2)
    return t[a:a + width].strip()


def _prev_nonempty(doc: Document, no: int) -> str:
    for i in range(no - 1, 0, -1):
        if re.search(r"[A-Za-zА-Яа-я]", doc.lines[i - 1].text):
            return doc.lines[i - 1].text.strip()
    return doc.lines[no - 1].text


def _dedupe(items):
    seen, out = set(), []
    for f in items:
        k = (f["criterion"], f["title"], f["lines"][0])
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def hints_text(findings: list[dict]) -> str:
    if not findings:
        return "(нет)"
    return "\n".join(f"- [L{f['lines'][0]}] {f['title']}: «{f['quote'][:100]}»" for f in findings)
