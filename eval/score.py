"""Оценка полноты агента по экспертному эталону.

python eval/score.py reports/report.json [eval/golden.json]

Эталонный дефект считается найденным, если хотя бы одно итоговое замечание агента
содержит любой из его «якорей» (в цитатах, заголовке или объяснении).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validator.document import normalize  # noqa: E402


def main():
    rep = json.loads(Path(sys.argv[1]).read_text("utf-8"))
    gold = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).with_name("golden.json")).read_text("utf-8"))
    texts = []
    for f in rep["findings"]:
        blob = " ".join(str(f.get(k) or "") for k in ("quote", "quote2", "title", "explanation", "recommendation"))
        blob += " " + " ".join(f"{a.get('quote') or ''} {a.get('title') or ''}" for a in f.get("also") or [])
        texts.append((f["id"], normalize(blob)))
    found = 0
    by_sev = {}
    print(f"{'ID':4} {'Сер.':9} {'Найден':8} Дефект")
    for g in gold["defects"]:
        hits = [fid for fid, t in texts if any(normalize(a) in t for a in g["anchors"])]
        ok = bool(hits)
        found += ok
        s = by_sev.setdefault(g["severity"], [0, 0])
        s[0] += ok
        s[1] += 1
        print(f"{g['id']:4} {g['severity']:9} {'✅ ' + ','.join(hits[:3]) if ok else '❌':8} {g['title'][:90]}")
    n = len(gold["defects"])
    print(f"\nПолнота: {found}/{n} = {found / n:.0%}  |  " +
          "  ".join(f"{k}: {v[0]}/{v[1]}" for k, v in by_sev.items()))
    print(f"Всего замечаний в отчёте: {len(rep['findings'])}")


if __name__ == "__main__":
    main()
