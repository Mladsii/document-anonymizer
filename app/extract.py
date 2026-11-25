import io
import zipfile
from pathlib import Path
from defusedxml import ElementTree


class DocumentError(ValueError):
    pass


def extract_text(data: bytes, filename: str, max_chars=300_000) -> str:
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".txt":
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = data.decode("cp1251")
            if "\x00" in text:
                raise DocumentError("Ожидается текстовый файл UTF-8 или Windows-1251.")
        elif suffix == ".docx":
            # Parse document body in XML order, including tables and hyperlink text.
            # Also include headers, footers, notes and comments, without transferring metadata.
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                entries = archive.infolist()
                if len(entries) > 2000 or sum(i.file_size for i in entries) > 50_000_000:
                    raise DocumentError("Слишком большой распакованный DOCX.")
                names = ["word/document.xml"] + sorted(
                    n for n in archive.namelist()
                    if n.startswith(("word/header", "word/footer", "word/footnotes", "word/endnotes", "word/comments")) and n.endswith(".xml")
                )
                paragraphs = []
                ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                for name in names:
                    root = ElementTree.fromstring(archive.read(name))
                    for paragraph in root.iter(ns + "p"):
                        fragments = []
                        for item in paragraph.iter():
                            if item.tag == ns + "t":
                                fragments.append(item.text or "")
                            elif item.tag == ns + "tab":
                                fragments.append("\t")
                            elif item.tag in {ns + "br", ns + "cr"}:
                                fragments.append("\n")
                        paragraphs.append("".join(fragments))
                text = "\n".join(paragraphs)
        elif suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise DocumentError("PDF защищён паролем. Загрузите незашифрованный файл.")
            if len(reader.pages) > 300:
                raise DocumentError("Допускается не более 300 страниц PDF.")
            parts = []
            for page in reader.pages:
                content = page.extract_text() or ""
                if not content.strip():
                    raise DocumentError("PDF загружен, но содержит скан без текстового слоя. Для таких файлов нужен OCR.")
                parts.append(content)
                if sum(map(len, parts)) > max_chars:
                    raise DocumentError("В документе слишком много текста.")
            text = "\n".join(parts)
        else:
            raise DocumentError("Поддерживаются TXT, DOCX и текстовый PDF. DOC и сканы нужно предварительно преобразовать.")
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("Не удалось прочитать документ. Проверьте формат и целостность файла.") from exc
    if not text.strip():
        raise DocumentError("В документе нет текста.")
    if len(text) > max_chars:
        raise DocumentError("Допускается не более 300 000 символов.")
    return text
