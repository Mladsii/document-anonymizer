"""Local Ollama entity extraction. Ollama receives document text only on localhost."""
import json
import os
import urllib.error
import urllib.request


class OllamaUnavailable(RuntimeError):
    pass


class OllamaNER:
    def __init__(self, endpoint=None, model=None, timeout=None):
        self.endpoint = (endpoint or os.getenv("OLLAMA_URL", "http://127.0.0.1:11343")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", "180"))

    def spans(self, text):
        # Structured extraction avoids asking the model to rewrite source text.
        prompt = (
            "Найди в тексте только персональные и корпоративные сущности, которые нужно скрыть: "
            "имена людей, организации, бренды и проекты, физические адреса, даты рождения. "
            "Не включай тайм-коды, метки Спикер A/B и обычные слова. "
            "Верни JSON строго по схеме {\"entities\":[{\"text\":\"точная подстрока из исходного текста\","
            "\"type\":\"PERSON|ORG|ADDRESS|BIRTHDATE\"}]}. "
            "Не добавляй пояснений. Если сущностей нет — пустой массив.\n\nТекст:\n" + text
        )
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 4096},
        }).encode()
        request = urllib.request.Request(self.endpoint + "/api/generate", data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
            answer = json.loads(result["response"])
            entities = answer.get("entities", [])
            spans = []
            categories = {"PERSON": "ИМЯ", "ORG": "КОМПАНИЯ", "ADDRESS": "АДРЕС", "BIRTHDATE": "ДАТА_РОЖДЕНИЯ"}
            cursor_by_text = {}
            for entity in entities:
                value, category = entity.get("text"), categories.get(entity.get("type"))
                if not isinstance(value, str) or not value or not category:
                    continue
                # Accept exact source substrings only. Never trust model-provided offsets.
                cursor = cursor_by_text.get(value, 0)
                start = text.find(value, cursor)
                if start < 0:
                    start = text.find(value)
                if start >= 0:
                    from app.engine import Span
                    spans.append(Span(start, start + len(value), category, 30))
                    cursor_by_text[value] = start + len(value)
            return spans
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise OllamaUnavailable("Ollama недоступна или вернула неверный ответ") from exc
