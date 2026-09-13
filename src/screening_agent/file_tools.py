"""Small, explicit file tools for a resume-oriented assistant.

The public functions in this module match the assignment contract.  They know
nothing about an LLM and can be called from a shell, a test, or another Python
program.  Workspace confinement belongs in the assistant layer because a plain
file utility should not silently change the meaning of an absolute path.

Every dict-returning tool uses the same top-level shape::

    {"ok": True, ...}
    {"ok": False, "error": {"code": "...", "message": "..."}}

`list_files` intentionally returns a list on success, as required by the brief.
It raises :class:`FileToolError` for invalid directories; the LLM dispatcher
converts that exception to the same structured error shape.
"""

from __future__ import annotations

import bisect
import codecs
import hashlib
import os
import re
import stat as stat_module
import tempfile
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from charset_normalizer import from_bytes
from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

SUPPORTED_RESUME_EXTENSIONS = frozenset({".pdf", ".txt", ".docx"})
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_DOCX_ENTRIES = 5_000
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_DOCX_COMPRESSION_RATIO = 200
MAX_SEARCH_MATCHES = 100
MAX_SEARCH_KEYWORD_CHARS = 256
SEARCH_CONTEXT_CHARS = 120


class FileToolError(Exception):
    """Expected file-tool failure with a stable, machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        filepath: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.filepath = filepath
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.filepath is not None:
            error["filepath"] = self.filepath
        if self.details:
            error["details"] = self.details
        return error


@dataclass(frozen=True)
class _Extraction:
    content: str
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]
    warnings: list[str]


def _error_result(error: FileToolError) -> dict[str, Any]:
    return {"ok": False, "error": error.as_dict()}


def _clean_path(value: str, *, field: str = "filepath") -> Path:
    if not isinstance(value, str):
        raise FileToolError(
            "invalid_argument",
            f"{field} must be a string",
            details={"field": field, "received_type": type(value).__name__},
        )
    if not value.strip():
        raise FileToolError(
            "invalid_argument", f"{field} cannot be empty", details={"field": field}
        )
    if "\x00" in value:
        raise FileToolError(
            "invalid_argument", f"{field} contains a NUL byte", details={"field": field}
        )
    return Path(value).expanduser()


def _iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'+.-]+\b", text, flags=re.UNICODE))


def _base_metadata(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()
    metadata: dict[str, Any] = {
        "name": path.name,
        "path": str(path.resolve()),
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified_at": _iso_utc(stat.st_mtime),
        "is_symlink": path.is_symlink(),
    }
    if include_hash:
        metadata["sha256"] = _sha256(path)
    return metadata


def _listing_metadata(path: Path, entry_stat: os.stat_result) -> dict[str, Any]:
    """Describe a directory entry without dereferencing its final symlink.

    A listing is often the first model-visible operation.  Resolving a link here
    would disclose the target's path, size, and timestamp before workspace policy
    has had a chance to inspect it.  `lstat` data keeps the public list useful
    while making that boundary explicit.
    """

    is_symlink = stat_module.S_ISLNK(entry_stat.st_mode)
    return {
        "name": path.name,
        "path": str(path.absolute()),
        "extension": path.suffix.lower(),
        "size_bytes": entry_stat.st_size,
        "modified_at": _iso_utc(entry_stat.st_mtime),
        "is_symlink": is_symlink,
        # Access to a link itself says nothing reliable about access to its
        # target, so do not follow it merely to populate convenience metadata.
        "readable": None if is_symlink else os.access(path, os.R_OK),
    }


def _validate_read_target(path: Path) -> None:
    if not path.exists():
        raise FileToolError("not_found", "File does not exist", filepath=str(path))
    if not path.is_file():
        raise FileToolError("not_a_file", "Path is not a file", filepath=str(path))

    extension = path.suffix.lower()
    if extension not in SUPPORTED_RESUME_EXTENSIONS:
        raise FileToolError(
            "unsupported_format",
            "Supported resume formats are PDF, TXT, and DOCX",
            filepath=str(path),
            details={"extension": extension or None},
        )

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise FileToolError(
            "file_too_large",
            f"File exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MiB safety limit",
            filepath=str(path),
            details={"size_bytes": size, "limit_bytes": MAX_FILE_BYTES},
        )


def _validate_pdf_signature(path: Path) -> None:
    with path.open("rb") as stream:
        signature = stream.read(5)
    if signature != b"%PDF-":
        raise FileToolError(
            "format_mismatch",
            "File extension is .pdf but the PDF signature is missing",
            filepath=str(path),
        )


def _validate_docx_package(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = {item.filename for item in entries}
            if len(names) != len(entries):
                raise FileToolError(
                    "corrupt_document",
                    "DOCX package contains duplicate archive names",
                    filepath=str(path),
                )
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise FileToolError(
                    "format_mismatch",
                    "The ZIP package is not a Word document",
                    filepath=str(path),
                )
            if len(entries) > MAX_DOCX_ENTRIES:
                raise FileToolError(
                    "unsafe_document",
                    "DOCX package contains too many entries",
                    filepath=str(path),
                    details={"entry_count": len(entries), "limit": MAX_DOCX_ENTRIES},
                )

            uncompressed = sum(item.file_size for item in entries)
            compressed = sum(item.compress_size for item in entries) or 1
            ratio = uncompressed / compressed
            if uncompressed > MAX_DOCX_UNCOMPRESSED_BYTES or ratio > MAX_DOCX_COMPRESSION_RATIO:
                raise FileToolError(
                    "unsafe_document",
                    "DOCX package failed the decompression safety check",
                    filepath=str(path),
                    details={
                        "uncompressed_bytes": uncompressed,
                        "compression_ratio": round(ratio, 2),
                    },
                )
    except zipfile.BadZipFile as exc:
        raise FileToolError(
            "corrupt_document", "DOCX package is not a valid ZIP file", filepath=str(path)
        ) from exc


def _validate_text_bytes(path: Path, data: bytes) -> None:
    if b"\x00" in data[:8192]:
        raise FileToolError(
            "binary_text_file",
            "TXT file contains NUL bytes and appears to be binary",
            filepath=str(path),
        )


def _extract_txt(path: Path) -> _Extraction:
    data = path.read_bytes()
    warnings: list[str] = []
    # UTF-32 starts with a UTF-16 prefix, so inspect the longest BOM first.
    encoding = next(
        (
            codec
            for bom, codec in (
                (codecs.BOM_UTF32_LE, "utf-32"),
                (codecs.BOM_UTF32_BE, "utf-32"),
                (codecs.BOM_UTF16_LE, "utf-16"),
                (codecs.BOM_UTF16_BE, "utf-16"),
                (codecs.BOM_UTF8, "utf-8-sig"),
            )
            if data.startswith(bom)
        ),
        "utf-8-sig",
    )
    if encoding == "utf-8-sig":
        _validate_text_bytes(path, data)
    try:
        content = data.decode(encoding)
    except UnicodeDecodeError as exc:
        if encoding != "utf-8-sig":
            raise FileToolError(
                "encoding_error", "Text has a malformed Unicode byte sequence", filepath=str(path)
            ) from exc
        best = from_bytes(data).best()
        if best is None or best.chaos > 0.1:
            raise FileToolError(
                "encoding_error",
                "Could not determine the TXT file encoding",
                filepath=str(path),
            ) from exc
        content = str(best)
        encoding = best.encoding or "unknown"
        warnings.append(f"Decoded non-UTF-8 text using {encoding}")
    if any(ord(character) < 32 and character not in "\n\r\t\f" for character in content):
        raise FileToolError(
            "binary_text_file",
            "Decoded text contains binary control characters",
            filepath=str(path),
        )
    if encoding == "utf-8-sig":
        encoding = "utf-8"
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content:
        warnings.append("The file is empty")
    return _Extraction(
        content=content,
        segments=[{"kind": "text", "index": 1, "label": "document", "text": content}],
        metadata={"extractor": "text", "encoding": encoding},
        warnings=warnings,
    )


def _extract_pdf(path: Path) -> _Extraction:
    _validate_pdf_signature(path)
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            unlocked: Any
            try:
                unlocked = reader.decrypt("")
            except Exception:  # pypdf raises several provider-specific errors
                unlocked = 0
            if not unlocked:
                raise FileToolError(
                    "encrypted_document",
                    "PDF is encrypted and cannot be read without a password",
                    filepath=str(path),
                )

        page_count = len(reader.pages)
        if page_count > MAX_PDF_PAGES:
            raise FileToolError(
                "document_too_long",
                f"PDF exceeds the {MAX_PDF_PAGES}-page safety limit",
                filepath=str(path),
                details={"page_count": page_count, "limit": MAX_PDF_PAGES},
            )

        segments: list[dict[str, Any]] = []
        blank_pages: list[int] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\r\n", "\n").strip()
            if not text:
                blank_pages.append(page_number)
            segments.append(
                {
                    "kind": "page",
                    "index": page_number,
                    "label": f"page {page_number}",
                    "text": text,
                }
            )
    except FileToolError:
        raise
    except Exception as exc:
        raise FileToolError(
            "corrupt_document",
            "PDF parser could not read the document",
            filepath=str(path),
            details={"reason": str(exc)},
        ) from exc

    content = "\n\n".join(segment["text"] for segment in segments if segment["text"])
    warnings: list[str] = []
    if blank_pages:
        warnings.append("No extractable text on page(s): " + ", ".join(map(str, blank_pages)))
    if not content:
        warnings.append("No extractable text found; the PDF may contain scanned images")
    return _Extraction(
        content=content,
        segments=segments,
        metadata={
            "extractor": "pypdf",
            "page_count": len(segments),
            "blank_page_count": len(blank_pages),
            "ocr_required": not bool(content),
        },
        warnings=warnings,
    )


def _iter_docx_blocks(document: DocxDocument) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in their true body order."""

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _cell_content(cell: Any) -> str:
    """Include nested tables in their true order and avoid merged-cell repeats."""
    parts: list[str] = []
    for block in cell.iter_inner_content():
        if isinstance(block, Paragraph):
            if block.text.strip():
                parts.append(block.text.strip())
        else:
            seen: set[Any] = set()
            for row in block.rows:
                for nested in row.cells:
                    identity = nested._tc
                    if identity not in seen:
                        seen.add(identity)
                        parts.append(_cell_content(nested))
    return " ; ".join(part for part in parts if part)


def _extract_docx(path: Path) -> _Extraction:
    _validate_docx_package(path)
    try:
        document = Document(str(path))
    except FileToolError:
        raise
    except Exception as exc:
        raise FileToolError(
            "corrupt_document",
            "Word parser could not read the document",
            filepath=str(path),
            details={"reason": str(exc)},
        ) from exc

    segments: list[dict[str, Any]] = []
    warnings: list[str] = []
    paragraph_count = 0
    table_count = 0

    # Word headers often contain contact details, so the extractor includes them.
    seen_header_footer: set[tuple[str, str]] = set()
    for section_number, section in enumerate(document.sections, start=1):
        for kind, part in (("header", section.header), ("footer", section.footer)):
            text = "\n".join(p.text.strip() for p in part.paragraphs if p.text.strip())
            key = (kind, text)
            if text and key not in seen_header_footer:
                seen_header_footer.add(key)
                segments.append(
                    {
                        "kind": kind,
                        "index": section_number,
                        "label": f"{kind} {section_number}",
                        "text": text,
                    }
                )

    block_number = 0
    for block in _iter_docx_blocks(document):
        block_number += 1
        if isinstance(block, Paragraph):
            paragraph_count += 1
            text = block.text.strip()
            if text:
                segments.append(
                    {
                        "kind": "paragraph",
                        "index": block_number,
                        "label": f"paragraph {block_number}",
                        "style": block.style.name if block.style is not None else None,
                        "text": text,
                    }
                )
            continue

        table_count += 1
        seen_cells: set[Any] = set()
        for row_number, row in enumerate(block.rows, start=1):
            cells = []
            for cell in row.cells:
                identity = cell._tc
                if identity not in seen_cells:
                    seen_cells.add(identity)
                    cells.append(_cell_content(cell))
            text = " | ".join(cell for cell in cells if cell)
            if text:
                segments.append(
                    {
                        "kind": "table_row",
                        "index": row_number,
                        "label": f"table {table_count}, row {row_number}",
                        "table_index": table_count,
                        "row_index": row_number,
                        "text": text,
                    }
                )

    content = "\n".join(segment["text"] for segment in segments if segment["text"])
    if not content:
        warnings.append("No extractable text found in paragraphs, tables, headers, or footers")
    return _Extraction(
        content=content,
        segments=segments,
        metadata={
            "extractor": "python-docx",
            "paragraph_count": paragraph_count,
            "table_count": table_count,
            "segment_count": len(segments),
        },
        warnings=warnings,
    )


def read_file(filepath: str) -> dict[str, Any]:
    """Read a PDF, TXT, or DOCX resume and return text with metadata.

    Expected file, format, encoding, permission, and parser failures are returned
    as data.  The response also includes provenance segments so a caller can tie
    a statement back to a PDF page or Word block.
    """

    try:
        path = _clean_path(filepath)
        _validate_read_target(path)
        extension = path.suffix.lower()
        extractors = {
            ".txt": _extract_txt,
            ".pdf": _extract_pdf,
            ".docx": _extract_docx,
        }
        try:
            extraction = extractors[extension](path)
        except (FileToolError, OSError):
            raise
        except Exception as exc:
            if extension in {".pdf", ".docx"}:
                raise FileToolError(
                    "corrupt_document",
                    "Document extraction failed after opening the file",
                    filepath=str(path),
                    details={"reason": str(exc)},
                ) from exc
            raise
        metadata = _base_metadata(path)
        metadata.update(extraction.metadata)
        metadata.update(
            {
                "character_count": len(extraction.content),
                "word_count": _word_count(extraction.content),
            }
        )
        return {
            "ok": True,
            "content": extraction.content,
            "metadata": metadata,
            "segments": extraction.segments,
            "warnings": extraction.warnings,
        }
    except FileToolError as exc:
        return _error_result(exc)
    except PermissionError:
        return _error_result(
            FileToolError("permission_denied", "Permission denied", filepath=str(filepath))
        )
    except OSError as exc:
        return _error_result(
            FileToolError(
                "io_error",
                "Operating system error while reading the file",
                filepath=str(filepath),
                details={"reason": str(exc)},
            )
        )
    except Exception as exc:  # Keep tool failures inside the tool boundary.
        return _error_result(
            FileToolError(
                "unexpected_error",
                "Unexpected error while reading the file",
                filepath=str(filepath),
                details={"reason": str(exc)},
            )
        )


def list_files(directory: str, extension: str | None = None) -> list[dict[str, Any]]:
    """List direct child files with deterministic metadata ordering.

    `extension` is case-insensitive and may be supplied with or without a dot.
    The function returns direct children only. Recursive traversal is outside
    this interface.
    """

    path = _clean_path(directory, field="directory")
    if not path.exists():
        raise FileToolError("not_found", "Directory does not exist", filepath=str(path))
    if not path.is_dir():
        raise FileToolError("not_a_directory", "Path is not a directory", filepath=str(path))

    normalized_extension: str | None = None
    if extension is not None:
        if not isinstance(extension, str) or not extension.strip():
            raise FileToolError(
                "invalid_extension",
                "extension must be a non-empty string or null",
                details={"extension": extension},
            )
        normalized_extension = extension.strip().lower()
        if not normalized_extension.startswith("."):
            normalized_extension = f".{normalized_extension}"

    files: list[dict[str, Any]] = []
    try:
        entries = sorted(path.iterdir(), key=lambda item: (item.name.casefold(), item.name))
        for item in entries:
            try:
                entry_stat = item.lstat()
            except FileNotFoundError:
                # A concurrent rename/removal should not invalidate the other
                # stable entries in the listing.
                continue
            is_file = stat_module.S_ISREG(entry_stat.st_mode)
            is_symlink = stat_module.S_ISLNK(entry_stat.st_mode)
            if not is_file and not is_symlink:
                continue
            if normalized_extension and item.suffix.lower() != normalized_extension:
                continue
            files.append(_listing_metadata(item, entry_stat))
    except PermissionError as exc:
        raise FileToolError(
            "permission_denied", "Permission denied while listing directory", filepath=str(path)
        ) from exc
    except OSError as exc:
        raise FileToolError(
            "io_error",
            "Operating system error while listing directory",
            filepath=str(path),
            details={"reason": str(exc)},
        ) from exc
    return files


def write_file(filepath: str, content: str) -> dict[str, Any]:
    """Atomically write UTF-8 text, creating parent directories as needed."""

    temp_path: Path | None = None
    try:
        path = _clean_path(filepath)
        if not isinstance(content, str):
            raise FileToolError(
                "invalid_argument",
                "content must be a string",
                filepath=str(path),
                details={"received_type": type(content).__name__},
            )
        if path.exists() and path.is_dir():
            raise FileToolError(
                "not_a_file", "Cannot write text to a directory", filepath=str(path)
            )

        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_temp_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temp_path = Path(raw_temp_path)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        temp_path = None

        # Persist the directory entry where the platform supports directory fsync.
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass

        metadata = _base_metadata(path)
        metadata.update({"encoding": "utf-8", "character_count": len(content)})
        return {
            "ok": True,
            "status": "overwritten" if existed else "created",
            "metadata": metadata,
        }
    except FileToolError as exc:
        return _error_result(exc)
    except PermissionError:
        return _error_result(
            FileToolError("permission_denied", "Permission denied", filepath=str(filepath))
        )
    except OSError as exc:
        return _error_result(
            FileToolError(
                "io_error",
                "Operating system error while writing the file",
                filepath=str(filepath),
                details={"reason": str(exc)},
            )
        )
    except Exception as exc:
        return _error_result(
            FileToolError(
                "unexpected_error",
                "Unexpected error while writing the file",
                filepath=str(filepath),
                details={"reason": str(exc)},
            )
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _segment_ranges(
    content: str, segments: list[dict[str, Any]]
) -> list[tuple[int, int, dict[str, Any]]]:
    ranges: list[tuple[int, int, dict[str, Any]]] = []
    cursor = 0
    for segment in segments:
        text = segment.get("text", "")
        if not text:
            continue
        start = content.find(text, cursor)
        if start < 0:
            continue
        end = start + len(text)
        location = {key: value for key, value in segment.items() if key != "text"}
        ranges.append((start, end, location))
        cursor = end
    return ranges


def _line_numbers_at_offsets(content: str, offsets: list[int]) -> dict[int, int]:
    """Map ordered character offsets to one-based lines with bounded memory."""

    line_numbers: dict[int, int] = {}
    line_number = 1
    cursor = 0
    for offset in sorted(set(offsets)):
        bounded_offset = min(max(0, offset), len(content))
        line_number += content.count("\n", cursor, bounded_offset)
        line_numbers[offset] = line_number
        cursor = bounded_offset
    return line_numbers


def search_in_file(filepath: str, keyword: str) -> dict[str, Any]:
    """Search extracted resume text literally and case-insensitively.

    Matches include a bounded character context, offsets, line spans, and source
    provenance such as a PDF page or DOCX table row.  At most 100 matches are
    retained while every occurrence is counted in a streaming pass.
    """

    if not isinstance(keyword, str) or not keyword:
        return _error_result(
            FileToolError(
                "invalid_keyword",
                "keyword must be a non-empty string",
                filepath=str(filepath),
            )
        )
    if len(keyword) > MAX_SEARCH_KEYWORD_CHARS:
        return _error_result(
            FileToolError(
                "invalid_keyword",
                f"keyword cannot exceed {MAX_SEARCH_KEYWORD_CHARS} characters",
                filepath=str(filepath),
                details={
                    "keyword_characters": len(keyword),
                    "limit": MAX_SEARCH_KEYWORD_CHARS,
                },
            )
        )

    result = read_file(filepath)
    if not result.get("ok"):
        return result

    content: str = result["content"]
    pattern = re.compile(re.escape(keyword), flags=re.IGNORECASE | re.UNICODE)
    match_count = 0
    selected_matches: list[re.Match[str]] = []
    for found in pattern.finditer(content):
        match_count += 1
        if len(selected_matches) < MAX_SEARCH_MATCHES:
            selected_matches.append(found)

    source_ranges = _segment_ranges(content, result.get("segments", []))
    source_starts = [item[0] for item in source_ranges]
    matches: list[dict[str, Any]] = []

    context_bounds = [
        (
            max(0, match.start() - SEARCH_CONTEXT_CHARS),
            min(len(content), match.end() + SEARCH_CONTEXT_CHARS),
        )
        for match in selected_matches
    ]
    line_offsets: list[int] = []
    for match, (context_start_offset, context_end_offset) in zip(
        selected_matches, context_bounds, strict=True
    ):
        line_offsets.extend(
            [
                match.start(),
                max(match.start(), match.end() - 1),
                context_start_offset,
                max(context_start_offset, context_end_offset - 1),
            ]
        )
    line_numbers = _line_numbers_at_offsets(content, line_offsets)

    for ordinal, (match, bounds) in enumerate(
        zip(selected_matches, context_bounds, strict=True), start=1
    ):
        context_start_offset, context_end_offset = bounds
        context = content[context_start_offset:context_end_offset]
        if context_start_offset > 0:
            context = "…" + context
        if context_end_offset < len(content):
            context += "…"

        end_match_offset = max(match.start(), match.end() - 1)
        end_context_offset = max(context_start_offset, context_end_offset - 1)
        start_line = line_numbers[match.start()]
        end_line = line_numbers[end_match_offset]
        context_start_line = line_numbers[context_start_offset]
        context_end_line = line_numbers[end_context_offset]

        location: dict[str, Any] | None = None
        if source_ranges:
            source_index = bisect.bisect_right(source_starts, match.start()) - 1
            if source_index >= 0:
                source_start, source_end, source_location = source_ranges[source_index]
                if source_start <= match.start() < source_end:
                    location = source_location

        matches.append(
            {
                "match_number": ordinal,
                "matched_text": match.group(0),
                "start_offset": match.start(),
                "end_offset": match.end(),
                "start_line": start_line,
                "end_line": end_line,
                "context_start_offset": context_start_offset,
                "context_end_offset": context_end_offset,
                "context_start_line": context_start_line,
                "context_end_line": context_end_line,
                "context": context,
                "location": location,
            }
        )

    return {
        "ok": True,
        "keyword": keyword,
        "filepath": result["metadata"]["path"],
        "match_count": match_count,
        "returned_match_count": len(matches),
        "truncated": match_count > MAX_SEARCH_MATCHES,
        "matches": matches,
        "source": {
            "name": result["metadata"]["name"],
            "extension": result["metadata"]["extension"],
            "sha256": result["metadata"]["sha256"],
        },
        "warnings": result.get("warnings", []),
    }


__all__ = [
    "FileToolError",
    "MAX_SEARCH_KEYWORD_CHARS",
    "SUPPORTED_RESUME_EXTENSIONS",
    "list_files",
    "read_file",
    "search_in_file",
    "write_file",
]
