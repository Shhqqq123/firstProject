from __future__ import annotations

import html
import io
import subprocess
import tokenize
from pathlib import Path
from typing import Iterable, List


SOFTWARE_NAME = "基于五项常规肿瘤标志物的乳腺健康智能筛查系统 V1.0"
LINES_PER_PAGE = 50
TOTAL_PAGES = 60
TOTAL_LINES = LINES_PER_PAGE * TOTAL_PAGES


def list_source_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in [root / "app.py", root / "main.py", root / "desktop_app.py"]:
        if p.exists():
            files.append(p)
    files.extend(sorted((root / "medical_system").glob("*.py")))
    files.extend(sorted((root / "scripts").glob("*.py")))
    return [p for p in files if p.is_file()]


def strip_comments_and_blanks(text: str) -> List[str]:
    parts: List[tuple[int, str]] = []
    token_stream = tokenize.generate_tokens(io.StringIO(text).readline)
    for tok_type, tok_str, *_ in token_stream:
        if tok_type == tokenize.COMMENT:
            continue
        parts.append((tok_type, tok_str))
    cleaned = tokenize.untokenize(parts)
    out: List[str] = []
    for ln in cleaned.splitlines():
        ln = ln.rstrip()
        if not ln.strip():
            continue
        out.append(ln)
    return out


def collect_lines(files: Iterable[Path]) -> List[str]:
    out: List[str] = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        out.extend(strip_comments_and_blanks(text))
    return out


def ensure_3000(lines: List[str]) -> List[str]:
    if len(lines) >= TOTAL_LINES:
        return lines[:TOTAL_LINES]
    return lines + ["pass"] * (TOTAL_LINES - len(lines))


def xml_run_text(text: str) -> str:
    esc = html.escape(text)
    return (
        '<w:r>'
        '<w:rPr>'
        '<w:rFonts w:ascii="DengXian" w:hAnsi="DengXian" w:eastAsia="宋体" w:cs="DengXian"/>'
        '<w:sz w:val="24"/><w:szCs w:val="24"/>'
        "</w:rPr>"
        f'<w:t xml:space="preserve">{esc}</w:t>'
        "</w:r>"
    )


def xml_para(text: str, align: str = "left") -> str:
    return (
        "<w:p>"
        f'<w:pPr><w:jc w:val="{align}"/></w:pPr>'
        f"{xml_run_text(text)}"
        "</w:p>"
    )


def xml_header_line(page_num: int) -> str:
    name = html.escape(SOFTWARE_NAME)
    return (
        "<w:p>"
        "<w:pPr>"
        '<w:tabs><w:tab w:val="right" w:pos="9062"/></w:tabs>'
        "</w:pPr>"
        "<w:r><w:rPr>"
        '<w:rFonts w:ascii="DengXian" w:hAnsi="DengXian" w:eastAsia="宋体" w:cs="DengXian"/>'
        '<w:sz w:val="24"/><w:szCs w:val="24"/>'
        "</w:rPr>"
        f'<w:t xml:space="preserve">{name}</w:t>'
        "</w:r>"
        "<w:r><w:tab/></w:r>"
        "<w:r><w:rPr>"
        '<w:rFonts w:ascii="DengXian" w:hAnsi="DengXian" w:eastAsia="宋体" w:cs="DengXian"/>'
        '<w:sz w:val="24"/><w:szCs w:val="24"/>'
        "</w:rPr>"
        f"<w:t>{page_num}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def build_markdown(lines: List[str]) -> str:
    parts: List[str] = ["# 软著源代码（60页）", ""]
    for page in range(TOTAL_PAGES):
        chunk = lines[page * LINES_PER_PAGE : (page + 1) * LINES_PER_PAGE]
        xml_block: List[str] = [xml_header_line(page + 1)]
        for code_line in chunk:
            xml_block.append(xml_para(code_line))

        parts.append("```{=openxml}")
        parts.extend(xml_block)
        parts.append("```")

        if page != TOTAL_PAGES - 1:
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

    src_files = list_source_files(root)
    if not src_files:
        raise RuntimeError("未找到源码文件。")

    raw_lines = collect_lines(src_files)
    final_lines = ensure_3000(raw_lines)

    md_path = out_dir / "softcopyright_60pages.md"
    docx_path = out_dir / "软著源码_60页_最终版.docx"
    md_path.write_text(build_markdown(final_lines), encoding="utf-8")

    cmd = f'pandoc "{md_path}" -o "{docx_path}"'
    subprocess.run(cmd, shell=True, check=True, cwd=root)

    print(f"源代码清洗后总行数: {len(raw_lines)}")
    print(f"文档代码行数: {len(final_lines)}")
    print(f"输出文件: {docx_path}")


if __name__ == "__main__":
    main()
