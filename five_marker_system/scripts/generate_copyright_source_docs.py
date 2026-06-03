from __future__ import annotations

import io
import tokenize
from pathlib import Path
from typing import Iterable, List


SOFTWARE_NAME = "基于五项常规肿瘤标志物的乳腺健康智能筛查系统 V1.0"
LINES_PER_PAGE = 50
PAGES_PER_DOC = 30
LINES_PER_DOC = LINES_PER_PAGE * PAGES_PER_DOC


def list_source_files(root: Path) -> List[Path]:
    candidates = [
        root / "app.py",
        root / "main.py",
        root / "desktop_app.py",
    ]
    candidates.extend(sorted((root / "medical_system").glob("*.py")))
    candidates.extend(sorted((root / "scripts").glob("*.py")))
    files = [p for p in candidates if p.exists() and p.is_file()]
    return files


def strip_python_comments_and_blanks(text: str) -> List[str]:
    out: List[str] = []
    stream = io.StringIO(text)
    token_stream = tokenize.generate_tokens(stream.readline)

    cleaned_parts: List[tuple[int, str]] = []
    for tok_type, tok_str, *_ in token_stream:
        if tok_type == tokenize.COMMENT:
            continue
        cleaned_parts.append((tok_type, tok_str))

    cleaned_text = tokenize.untokenize(cleaned_parts)
    for line in cleaned_text.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        out.append(line)
    return out


def collect_clean_code_lines(files: Iterable[Path], root: Path) -> List[str]:
    lines: List[str] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        lines.append(f"# ===== File: {rel} =====")
        text = path.read_text(encoding="utf-8", errors="ignore")
        file_lines = strip_python_comments_and_blanks(text)
        lines.extend(file_lines)
    return lines


def pad_or_trim(lines: List[str], size: int) -> List[str]:
    if len(lines) >= size:
        return lines[:size]
    padded = list(lines)
    padded.extend(["pass"] * (size - len(lines)))
    return padded


def to_markdown_pages(lines: List[str], title: str) -> str:
    chunks = [lines[i : i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]
    parts: List[str] = [f"# {title}", ""]
    for page_idx, chunk in enumerate(chunks, start=1):
        parts.append(f"**{SOFTWARE_NAME}**")
        parts.append(f"第 {page_idx} 页")
        parts.append("")
        parts.append("```python")
        parts.extend(chunk)
        parts.append("```")
        if page_idx != len(chunks):
            parts.append("")
            parts.append("```{=openxml}")
            parts.append("<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>")
            parts.append("```")
            parts.append("")
    return "\n".join(parts)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "RelevantDocuments"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_source_files(root)
    if not files:
        raise RuntimeError("No source files found.")

    all_lines = collect_clean_code_lines(files, root)
    if not all_lines:
        raise RuntimeError("No code lines collected after cleaning.")

    front_lines = pad_or_trim(all_lines[:LINES_PER_DOC], LINES_PER_DOC)
    back_lines = pad_or_trim(all_lines[-LINES_PER_DOC:], LINES_PER_DOC)

    front_md = out_dir / "copyright_source_front30.md"
    back_md = out_dir / "copyright_source_back30.md"
    front_docx = out_dir / "copyright_source_front30.docx"
    back_docx = out_dir / "copyright_source_back30.docx"

    front_md.write_text(to_markdown_pages(front_lines, "软著源代码 前30页"), encoding="utf-8")
    back_md.write_text(to_markdown_pages(back_lines, "软著源代码 后30页"), encoding="utf-8")

    ref_doc = out_dir / "soft_code_format_copy.docx"
    pandoc = "pandoc"
    if ref_doc.exists():
        cmd_front = f'{pandoc} "{front_md}" -o "{front_docx}" --reference-doc="{ref_doc}"'
        cmd_back = f'{pandoc} "{back_md}" -o "{back_docx}" --reference-doc="{ref_doc}"'
    else:
        cmd_front = f'{pandoc} "{front_md}" -o "{front_docx}"'
        cmd_back = f'{pandoc} "{back_md}" -o "{back_docx}"'

    import subprocess

    subprocess.run(cmd_front, shell=True, check=True, cwd=root)
    subprocess.run(cmd_back, shell=True, check=True, cwd=root)

    print(f"Collected cleaned lines: {len(all_lines)}")
    print(f"Front: {front_docx}")
    print(f"Back: {back_docx}")


if __name__ == "__main__":
    main()
