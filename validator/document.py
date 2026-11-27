"""Чтение входного документа (PDF, DOCX, MD, TXT, HTML) и привязка цитат к строкам."""
from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Line:
    no: int          # сквозной номер строки (1..N)
    page: int | None  # номер страницы для PDF
    text: str


@dataclass
class Document:
    path: Path
    fmt: str
    lines: list[Line]
    sha256: str
    title: str = ""
    _norm_cache: tuple | None = field(default=None, repr=False)

    # ---------- представление для LLM ----------
    def numbered_text(self) -> str:
        """Текст с номерами строк: модель ссылается на [L12]."""
        return "\n".join(f"[L{l.no}] {l.text}" for l in self.lines if l.text.strip())

    def location(self, start: int, end: int | None = None) -> str:
        end = end or start
        span = f"строка {start}" if start == end else f"строки {start}–{end}"
        page = self.lines[start - 1].page
        return f"с. {page}, {span}" if page is not None else span

    # ---------- проверка цитат ----------
    def _normalized(self):
        if self._norm_cache is None:
            chars, owners = [], []
            for l in self.lines:
                n = normalize(l.text)
                if not n:
                    continue
                for ch in n + " ":
                    chars.append(ch)
                    owners.append(l.no)
            self._norm_cache = ("".join(chars), owners)
        return self._norm_cache

    def locate(self, quote: str, near: int | None = None) -> tuple[str, int | None, int | None]:
        """Ищет цитату в документе.

        Возвращает (status, line_start, line_end), где status:
        exact — найдена дословно (с точностью до пробелов/регистра/разметки),
        fuzzy — найдена близкая формулировка (≥ 85% совпадения),
        missing — в документе такой цитаты нет (вероятная галлюцинация модели).
        """
        q = normalize(quote)
        if len(q) < 3:
            return "missing", None, None
        text, owners = self._normalized()
        hits, idx = [], text.find(q)
        while idx >= 0:
            hits.append((owners[idx], owners[min(idx + len(q) - 1, len(owners) - 1)]))
            idx = text.find(q, idx + 1)
        if hits:
            if near is not None:
                hits.sort(key=lambda h: abs(h[0] - near))
            return "exact", hits[0][0], hits[0][1]
        # нечёткий поиск по окнам из 1–4 строк
        best = (0.0, None, None)
        normed = [(l.no, normalize(l.text)) for l in self.lines if normalize(l.text)]
        for i in range(len(normed)):
            acc = ""
            for j in range(i, min(i + 4, len(normed))):
                acc = (acc + " " + normed[j][1]).strip()
                if len(acc) < len(q) * 0.6:
                    continue
                sm = difflib.SequenceMatcher(None, q, acc, autojunk=False)
                m = sm.find_longest_match(0, len(q), 0, len(acc))
                ratio = m.size / len(q)
                if ratio < 0.5:
                    # частичное совпадение по самому длинному блоку слишком мало
                    ratio = max(ratio, _partial_ratio(q, acc))
                if ratio > best[0]:
                    best = (ratio, normed[i][0], normed[j][0])
                if len(acc) > len(q) * 2.5:
                    break
        if best[0] >= 0.85:
            return "fuzzy", best[1], best[2]
        # слова цитаты идут по порядку, но перемешаны с соседними колонками (таблицы в PDF)
        hit = self._subsequence(q, near)
        if hit:
            return "fuzzy", hit[0], hit[1]
        return "missing", None, None

    def _subsequence(self, q: str, near: int | None):
        qt = q.split()
        if len(qt) < 3:
            return None
        toks = [(w, l.no) for l in self.lines for w in normalize(l.text).split()]
        max_span = len(qt) * 3 + 6
        hits = []
        for i, (w, _) in enumerate(toks):
            if w != qt[0]:
                continue
            k, j = 1, i + 1
            while k < len(qt) and j < len(toks) and j - i <= max_span:
                if toks[j][0] == qt[k]:
                    k += 1
                j += 1
            if k == len(qt):
                hits.append((toks[i][1], toks[j - 1][1]))
        if not hits:
            return None
        if near is not None:
            hits.sort(key=lambda h: abs(h[0] - near))
        return hits[0]


def _partial_ratio(q: str, s: str) -> float:
    if len(s) <= len(q):
        return difflib.SequenceMatcher(None, q, s, autojunk=False).ratio()
    best = 0.0
    step = max(1, len(q) // 4)
    for k in range(0, len(s) - len(q) + 1, step):
        r = difflib.SequenceMatcher(None, q, s[k:k + len(q)], autojunk=False).ratio()
        best = max(best, r)
    return best


_MD = re.compile(r"[*_`#|>]+")


def normalize(s: str) -> str:
    s = s.replace(" ", " ").replace("ё", "е").replace("Ё", "Е")
    s = s.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"').replace("„", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = _MD.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


# ---------------- загрузчики ----------------

def load(path: str | Path) -> Document:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p}")
    ext = p.suffix.lower()
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if ext == ".pdf":
        lines, fmt = _load_pdf(p), "PDF"
    elif ext == ".docx":
        lines, fmt = _load_docx(p), "DOCX"
    elif ext in (".html", ".htm"):
        lines, fmt = _plain_lines(_html_to_text(raw.decode("utf-8", "replace"))), "HTML"
    elif ext in (".md", ".markdown", ".txt", ".rst", ""):
        lines, fmt = _plain_lines(raw.decode("utf-8-sig", "replace")), "Markdown" if ext.startswith(".m") else "Text"
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}. Поддерживаются PDF, DOCX, MD, TXT, HTML.")
    if not any(l.text.strip() for l in lines):
        raise ValueError("Документ пуст или текст не извлекается (скан без текстового слоя?).")
    doc = Document(path=p, fmt=fmt, lines=lines, sha256=sha)
    doc.title = _guess_title(doc)
    return doc


def _plain_lines(text: str, page: int | None = None, start: int = 1) -> list[Line]:
    out = []
    for i, t in enumerate(text.splitlines()):
        out.append(Line(no=start + i, page=page, text=t.rstrip()))
    return out


def _load_pdf(p: Path) -> list[Line]:
    lines: list[Line] = []
    try:
        import pdfplumber
        with pdfplumber.open(str(p)) as pdf:
            for pi, page in enumerate(pdf.pages, start=1):
                lines.extend(_plain_lines(_pdf_page_text(page), page=pi, start=len(lines) + 1))
    except ImportError:
        from pypdf import PdfReader
        for pi, page in enumerate(PdfReader(str(p)).pages, start=1):
            lines.extend(_plain_lines(page.extract_text() or "", page=pi, start=len(lines) + 1))
    return lines


def _pdf_page_text(page) -> str:
    """Текст страницы, где таблицы собраны построчно «| ячейка | ячейка |», а не вперемешку."""
    try:
        tables = page.find_tables()
    except Exception:
        tables = []
    if not tables:
        return page.extract_text() or ""
    boxes = [t.bbox for t in tables]

    def inside(obj):
        cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
        return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)

    items = []  # (top, text)
    rest = page.filter(lambda o: not (o.get("object_type") == "char" and inside(o)))
    for ln in rest.extract_text_lines():
        items.append((ln["top"], ln["text"]))
    for t in tables:
        for ri, row in enumerate(t.extract()):
            cells = [re.sub(r"\s+", " ", c or "").strip() for c in row]
            items.append((t.bbox[1] + ri * 0.01, "| " + " | ".join(cells) + " |"))
    items.sort(key=lambda x: x[0])
    return "\n".join(t for _, t in items)


def _load_docx(p: Path) -> list[Line]:
    import docx  # python-docx
    d = docx.Document(str(p))
    texts: list[str] = []
    # абзацы и таблицы в порядке следования в документе
    body = d.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            para = docx.text.paragraph.Paragraph(child, d)
            style = (para.style.name or "").lower() if para.style is not None else ""
            prefix = "#" * int(style[-1]) + " " if style.startswith("heading") and style[-1:].isdigit() else ""
            texts.append(prefix + para.text)
        elif tag == "tbl":
            table = docx.table.Table(child, d)
            for row in table.rows:
                texts.append("| " + " | ".join(c.text.strip() for c in row.cells) + " |")
    return _plain_lines("\n".join(texts))


def _html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", "", html)
    html = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</tr>|</h\d>", "\n", html)
    html = re.sub(r"(?i)</t[dh]>", " | ", html)
    text = re.sub(r"<[^>]+>", "", html)
    import html as h
    return h.unescape(text)


def _guess_title(doc: Document) -> str:
    for l in doc.lines:
        t = l.text.strip()
        if t:
            return t.lstrip("#").strip()[:150]
    return doc.path.name
