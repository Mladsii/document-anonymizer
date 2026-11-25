"""Offset-based replacement: rules and NER see original text; values never leave memory."""
import json
import ipaddress
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    category: str
    priority: int = 0
    canonical: str | None = None


FLAGS = re.I | re.U
DOMAIN = r"(?:[a-zа-яё0-9](?:[a-zа-яё0-9-]{0,61}[a-zа-яё0-9])?\.)+(?:[a-zа-яё]{2,63}|xn--[a-z0-9-]+)"
RULES = [
    ("EMAIL", rf"(?<![\w.+-])[\w.!#$%&'*+/=?^`{{|}}~-]{{1,64}}@{DOMAIN}", 100),
    ("ССЫЛКА", r"(?:https?://|ftp://|www\.)[^\s<>\"«»]+", 90),
    ("ССЫЛКА", rf"(?<![\w@-]){DOMAIN}(?::\d{{1,5}})?(?:/[^\s<>\"«»]*)?", 80),
    ("IP", r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w|\.\d)", 75),
    ("IP", r"(?<![\w:])(?:[a-f0-9]{0,4}:){2,7}[a-f0-9]{0,4}(?![\w:])", 75),
    ("МЕССЕНДЖЕР", r"(?<![\w@])@[a-z0-9_]{3,32}\b", 70),
    ("ПАСПОРТ", r"(?<!\d)\d{2}[ \t]?\d{2}[ \t]+\d{6}(?!\d)", 70),
    ("СНИЛС", r"(?<!\d)\d{3}-\d{3}-\d{3}[ \t-]?\d{2}(?!\d)", 70),
    ("КАРТА", r"(?<!\d)(?:\d{4}[ \t-]){3}\d{4}(?!\d)", 65),
    ("ТЕЛЕФОН", r"(?<!\w)(?:\+7|8)[ \t(-]*\d{3}[ \t)\-]*\d{3}[ \t-]*\d{2}[ \t-]*\d{2}(?!\d)", 60),
    ("РЕКВИЗИТЫ", r"(?<!\d)(?:\d{20}|\d{15}|\d{13}|\d{12}|\d{10})(?!\d)", 40),
    ("КОМПАНИЯ", r"\b(?:ООО|ОАО|ЗАО|ПАО|АО|ГК|ИП)[ \t]+[«\"][^»\"\n]+[»\"]", 85),
]
# Only the captured value is hidden, retaining the field label.
FIELDS = [
    ("РЕКВИЗИТЫ", r"\b(?:ИНН|КПП|ОГРН(?:ИП)?|БИК|р/с|к/с|расч[её]тный сч[её]т|корреспондентский сч[её]т)[ \t:№=-]*([\d][\d \t-]{4,30}\d)"),
    ("ПАСПОРТ", r"\b(?:код подразделения)[ \t:=-]*(\d{3}-?\d{3})"),
    ("ПАСПОРТ", r"\b(?:серия паспорта|паспорт[ \t:]*серия)[ \t:№=-]*(\d{2}[ \t]?\d{2})"),
    ("ПАСПОРТ", r"\b(?:номер паспорта|паспорт[ \t:]*номер)[ \t:№=-]*(\d{6})"),
    ("ПАСПОРТ", r"\b(?:кем выдан|паспорт выдан)[ \t:=-]*([^\n;]+)"),
    ("ПАСПОРТ", r"\b(?:дата выдачи|когда выдан)[ \t:=-]*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"),
    ("ДАТА_РОЖДЕНИЯ", r"\b(?:дата рождения|родился|родилась|д[.]р[.])[ \t:=-]*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"),
    ("АДРЕС", r"\b(?:юридический адрес|фактический адрес|адрес доставки|адрес проживания|адрес регистрации)[ \t:=-]*([^\n;]+)"),
]


def normalize(value):
    return " ".join(value.casefold().replace("ё", "е").split())


class NatashaNER:
    def __init__(self):
        from natasha import Segmenter, NewsEmbedding, NewsNERTagger
        self.segmenter = Segmenter()
        self.tagger = NewsNERTagger(NewsEmbedding())

    def spans(self, text):
        from natasha import Doc
        # Line boundaries preserve offsets and avoid inference on a whole huge document.
        offset = 0
        for line in text.splitlines(keepends=True):
            for start in range(0, len(line), 6000):
                part = line[start:start + 6000]
                doc = Doc(part)
                doc.segment(self.segmenter)
                doc.tag_ner(self.tagger)
                for item in doc.spans:
                    category = {"PER": "ИМЯ", "ORG": "КОМПАНИЯ", "LOC": "АДРЕС"}.get(item.type)
                    if category:
                        yield Span(offset + start + item.start, offset + start + item.stop, category, 20)
            offset += len(line)


class Anonymizer:
    def __init__(self, rules_path: Path, ner=None, llm=None):
        self.ner = ner
        self.llm = llm
        self.rules = [(cat, re.compile(pattern, FLAGS), priority) for cat, pattern, priority in RULES]
        self.fields = [(cat, re.compile(pattern, FLAGS)) for cat, pattern in FIELDS]
        self.aliases = []
        for group in json.loads(rules_path.read_text(encoding="utf-8"))["aliases"]:
            if not re.fullmatch(r"[А-ЯA-Z_]+", group["category"]):
                raise ValueError("Invalid alias category")
            for value in group["values"]:
                if not value.strip():
                    raise ValueError("Empty alias")
                self.aliases.append((re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)", FLAGS), group))

    def anonymize(self, text):
        spans = []
        for category, pattern, priority in self.rules:
            for match in pattern.finditer(text):
                if category == "IP":
                    try:
                        ipaddress.ip_address(match.group())
                    except ValueError:
                        continue
                end = match.end()
                if category == "ССЫЛКА":
                    end = match.start() + len(match.group().rstrip(".,;:!?)]}"))
                spans.append(Span(match.start(), end, category, priority))
        for category, pattern in self.fields:
            for m in pattern.finditer(text):
                start, end = m.span(1)
                end = start + len(m.group(1).rstrip(" \t.,"))
                spans.append(Span(start, end, category, 110))
        for pattern, group in self.aliases:
            spans.extend(Span(*m.span(), group["category"], 95, group["canonical"]) for m in pattern.finditer(text))
        if self.ner:
            spans.extend(self.ner.spans(text))
        if self.llm:
            spans.extend(self.llm.spans(text))
        # Never partially reveal a matched entity: union overlapping spans.
        # Preserve labels and timestamps even if NER classified them as entities.
        protected = [m.span() for m in re.finditer(r"\[[А-ЯA-Z_]+(?:_\d+)?\]|\bСпикер[ \t]+[A-ZА-Я]\b|\b\d{1,2}:\d{2}(?::\d{2})?\b", text)]
        spans = [s for s in spans if s.priority != 20 or not any(s.start < b and s.end > a for a, b in protected)]
        spans.sort(key=lambda s: (s.start, s.end))
        merged = []
        for span in spans:
            if merged and span.start < merged[-1].end:
                old = merged.pop()
                winner = max((old, span), key=lambda s: (s.priority, s.end - s.start))
                merged.append(Span(old.start, max(old.end, span.end), winner.category, winner.priority, winner.canonical))
            else:
                merged.append(span)
        counters, mapping, chunks, cursor = {}, {}, [], 0
        for span in merged:
            value = span.canonical or normalize(text[span.start:span.end])
            key = (span.category, value)
            if key not in mapping:
                counters[span.category] = counters.get(span.category, 0) + 1
                mapping[key] = f"[{span.category}_{counters[span.category]}]"
            chunks.extend((text[cursor:span.start], mapping[key]))
            cursor = span.end
        chunks.append(text[cursor:])
        return "".join(chunks)
