from pathlib import Path
from app.engine import Anonymizer, Span

RULES = Path(__file__).resolve().parents[1] / "config/rules.json"


def test_network_identifiers_and_punctuation():
    engine = Anonymizer(RULES)
    result = engine.anonymize("Почта me@sub.example.org, сайт example.org и https://example.org/a?q=1. @employee 192.0.2.1")
    assert "example" not in result
    assert "employee" not in result
    assert "192.0.2.1" not in result
    assert "[EMAIL_1], сайт" in result
    assert "[ССЫЛКА_2]." in result


def test_aliases_numbers_and_unchanged_timestamps():
    engine = Anonymizer(RULES)
    result = engine.anonymize("00:10 Спикер A: ООО «Пример» и ООО «Пример». +7 (999) 123-45-67. Бюджет 5000 рублей.")
    assert result.count("[КОМПАНИЯ_1]") == 2
    assert "00:10 Спикер A:" in result
    assert "[ТЕЛЕФОН_1]. Бюджет 5000 рублей." in result


def test_requisites_and_passport():
    engine = Anonymizer(RULES)
    result = engine.anonymize("КПП: 123456789; БИК: 987654321; Паспорт 12 34 567890; дата рождения: 01.02.1990")
    for value in ("123456789", "987654321", "567890", "01.02.1990"):
        assert value not in result


def test_ner_overlap_never_exposes_tail():
    class FakeNER:
        def spans(self, text):
            yield Span(0, len(text), "КОМПАНИЯ", 20)
    assert Anonymizer(RULES, FakeNER()).anonymize("ООО «Тест»") == "[КОМПАНИЯ_1]"


def test_document_local_mapping_and_existing_placeholders():
    engine = Anonymizer(RULES)
    assert engine.anonymize("a@example.org a@example.org").count("[EMAIL_1]") == 2
    assert engine.anonymize("b@example.org") == "[EMAIL_1]"
    assert engine.anonymize("[КОМПАНИЯ_1] 00:10 Спикер A") == "[КОМПАНИЯ_1] 00:10 Спикер A"


def test_ipv4_sentence_end_ipv6_and_url_timestamp():
    engine = Anonymizer(RULES)
    result = engine.anonymize("IP 192.0.2.15. IPv6 2001:db8::1. https://example.org/12:30")
    assert result == "IP [IP_1]. IPv6 [IP_2]. [ССЫЛКА_1]"
