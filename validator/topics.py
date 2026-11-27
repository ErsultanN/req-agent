import re

CHK = re.compile(r"Да\s*/\s*Нет")
TOPICS = [
    ("Схема отката", r"откат|rollback|feature[ -]?flag",
     "Не описано, как откатить изменение или выключить функцию."),
    ("Требования ИБ", r"информационн\w+ безопасн|PCI|маскир\w+|шифрован\w+",
     "Не заданы требования ИБ: защита карточных данных, маскирование в логах и URL."),
    ("Мониторинг", r"мониторинг|алерт|журнал\w*|логирован\w*",
     "Не заданы мониторинг, алерты и журналирование."),
]


def missing_topics(doc, make):
    res = []
    nfr = next((l.no for l in doc.lines if re.search(r"нефункциональн", l.text, re.I)), None)
    marks = [l.no for l in doc.lines if CHK.search(l.text)]
    def in_chk(no):
        return len(marks) >= 3 and marks[0] - 3 <= no <= marks[-1] + 3
    for title, pat, why in TOPICS:
        hits = [l for l in doc.lines if re.search(pat, l.text, re.I)]
        if [l for l in hits if not (CHK.search(l.text) or in_chk(l.no))]:
            continue
        if hits:
            res.append(make(doc, hits[0].no, "set_complete", "minor",
                            title + ": тема только в шаблоне чек-листа",
                            why + " Тема есть лишь как пустой пункт чек-листа.",
                            "Добавить раздел с требованиями.", scope="set"))
        elif nfr:
            res.append(make(doc, nfr, "set_complete", "minor", title + ": отсутствуют",
                            why, "Добавить требования в НФТ.", scope="set"))
    return res
