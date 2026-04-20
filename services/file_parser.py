"""
文件解析模块 —— 将用户上传的各种格式文件统一提取为文本 + 图片。

支持的格式：
  图片: .png, .jpg, .jpeg, .bmp, .gif, .webp
  PDF:  .pdf (通过 pymupdf 提取文本与内嵌图片)
  Word: .docx (通过 python-docx 提取文本与图片)
  Excel: .xlsx, .xls (通过 openpyxl 提取表格数据)
  PPT:  .pptx (通过 python-pptx 提取文本与图片)
  纯文本: .txt, .md, .csv, .json, .log
"""

from __future__ import annotations

import os
import logging
import tempfile
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log"}


@dataclass
class FileParseResult:
    """文件解析的统一返回结构。"""
    text: str = ""
    images: list[str] = field(default_factory=list)  # 图片文件路径列表


def parse_files(file_paths: list[str]) -> FileParseResult:
    """批量解析多个文件，合并结果。"""
    combined = FileParseResult()
    for path in file_paths:
        result = parse_file(path)
        if result.text:
            if combined.text:
                combined.text += "\n\n"
            combined.text += result.text
        combined.images.extend(result.images)
    return combined


def parse_file(file_path: str) -> FileParseResult:
    """解析单个文件，根据扩展名分发到对应解析器。"""
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return FileParseResult(text=f"[错误：文件不存在 {file_path}]")

    ext = os.path.splitext(file_path)[1].lower()
    file_path = os.path.abspath(file_path)

    if ext in IMAGE_EXTENSIONS:
        return _parse_image(file_path)
    elif ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext == ".docx":
        return _parse_docx(file_path)
    elif ext in (".xlsx", ".xls"):
        return _parse_xlsx(file_path)
    elif ext == ".pptx":
        return _parse_pptx(file_path)
    elif ext in TEXT_EXTENSIONS:
        return _parse_text(file_path)
    else:
        logger.warning(f"不支持的文件格式: {ext}，尝试作为纯文本读取: {file_path}")
        return _parse_text(file_path)


# ── 各格式解析器 ──────────────────────────────────────────────────────────────


def _parse_image(file_path: str) -> FileParseResult:
    """图片文件直接返回路径，不提取文本。"""
    logger.info(f"📷 加载图片: {file_path}")
    return FileParseResult(images=[file_path])


def _parse_text(file_path: str) -> FileParseResult:
    """读取纯文本文件。"""
    logger.info(f"📄 读取文本文件: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return FileParseResult(text=content)
    except UnicodeDecodeError:
        # 尝试其他编码
        with open(file_path, "r", encoding="gbk", errors="replace") as f:
            content = f.read().strip()
        return FileParseResult(text=content)


def _parse_pdf(file_path: str) -> FileParseResult:
    """使用 pymupdf 提取 PDF 中的文本和图片。"""
    try:
        import fitz  # pymupdf
    except ImportError:
        logger.error("缺少 pymupdf 库，无法解析 PDF。请运行: pip install pymupdf")
        return FileParseResult(text="[错误：缺少 pymupdf 库，无法解析 PDF]")

    logger.info(f"📕 解析 PDF: {file_path}")
    text_parts = []
    images = []
    tmp_dir = tempfile.mkdtemp(prefix="file_parser_pdf_")

    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            # 提取文本
            page_text = page.get_text().strip()
            if page_text:
                text_parts.append(f"[第{page_num + 1}页]\n{page_text}")

            # 提取图片
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:  # CMYK → RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_path = os.path.join(tmp_dir, f"pdf_p{page_num + 1}_img{img_idx + 1}.png")
                    pix.save(img_path)
                    images.append(img_path)
                    logger.info(f"  提取图片: {img_path}")
                except Exception as e:
                    logger.warning(f"  提取 PDF 图片失败 (页{page_num + 1}, 图{img_idx + 1}): {e}")
        doc.close()
    except Exception as e:
        logger.error(f"PDF 解析失败: {e}")
        return FileParseResult(text=f"[错误：PDF 解析失败 - {e}]")

    return FileParseResult(text="\n\n".join(text_parts), images=images)


def _parse_docx(file_path: str) -> FileParseResult:
    """使用 python-docx 提取 Word 文档中的文本和图片。"""
    try:
        from docx import Document
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
    except ImportError:
        logger.error("缺少 python-docx 库，无法解析 Word 文档。请运行: pip install python-docx")
        return FileParseResult(text="[错误：缺少 python-docx 库，无法解析 Word 文档]")

    logger.info(f"📘 解析 Word: {file_path}")
    text_parts = []
    images = []
    tmp_dir = tempfile.mkdtemp(prefix="file_parser_docx_")

    try:
        doc = Document(file_path)

        # 提取文本段落
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text.strip())

        # 提取表格
        for table_idx, table in enumerate(doc.tables):
            table_text = f"[表格 {table_idx + 1}]\n"
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                table_text += " | ".join(cells) + "\n"
            text_parts.append(table_text)

        # 提取图片
        img_idx = 0
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                img_idx += 1
                try:
                    img_data = rel.target_part.blob
                    ext = os.path.splitext(rel.target_ref)[1] or ".png"
                    img_path = os.path.join(tmp_dir, f"docx_img{img_idx}{ext}")
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    images.append(img_path)
                    logger.info(f"  提取图片: {img_path}")
                except Exception as e:
                    logger.warning(f"  提取 Word 图片失败 (图{img_idx}): {e}")

    except Exception as e:
        logger.error(f"Word 解析失败: {e}")
        return FileParseResult(text=f"[错误：Word 解析失败 - {e}]")

    return FileParseResult(text="\n\n".join(text_parts), images=images)


def _parse_xlsx(file_path: str) -> FileParseResult:
    """使用 openpyxl 提取 Excel 表格数据为格式化文本。"""
    try:
        import openpyxl
    except ImportError:
        logger.error("缺少 openpyxl 库，无法解析 Excel。请运行: pip install openpyxl")
        return FileParseResult(text="[错误：缺少 openpyxl 库，无法解析 Excel]")

    logger.info(f"📊 解析 Excel: {file_path}")
    text_parts = []

    try:
        wb = openpyxl.load_workbook(file_path, data_only=True)
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            sheet_text = f"[工作表: {sheet_name}]\n"
            for row in ws.iter_rows(values_only=True):
                cells = [str(cell) if cell is not None else "" for cell in row]
                sheet_text += " | ".join(cells) + "\n"
            text_parts.append(sheet_text)
        wb.close()
    except Exception as e:
        logger.error(f"Excel 解析失败: {e}")
        return FileParseResult(text=f"[错误：Excel 解析失败 - {e}]")

    return FileParseResult(text="\n\n".join(text_parts))


def _parse_pptx(file_path: str) -> FileParseResult:
    """使用 python-pptx 提取 PPT 中的文本和图片。"""
    try:
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE
    except ImportError:
        logger.error("缺少 python-pptx 库，无法解析 PPT。请运行: pip install python-pptx")
        return FileParseResult(text="[错误：缺少 python-pptx 库，无法解析 PPT]")

    logger.info(f"📙 解析 PPT: {file_path}")
    text_parts = []
    images = []
    tmp_dir = tempfile.mkdtemp(prefix="file_parser_pptx_")
    img_idx = 0

    try:
        prs = Presentation(file_path)
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_texts = []
            for shape in slide.shapes:
                # 提取文本
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            slide_texts.append(text)

                # 提取表格
                if shape.has_table:
                    table = shape.table
                    for row in table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        slide_texts.append(" | ".join(cells))

                # 提取图片
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    img_idx += 1
                    try:
                        img_blob = shape.image.blob
                        ext = shape.image.content_type.split("/")[-1]
                        if ext == "jpeg":
                            ext = "jpg"
                        img_path = os.path.join(tmp_dir, f"pptx_s{slide_num}_img{img_idx}.{ext}")
                        with open(img_path, "wb") as f:
                            f.write(img_blob)
                        images.append(img_path)
                        logger.info(f"  提取图片: {img_path}")
                    except Exception as e:
                        logger.warning(f"  提取 PPT 图片失败 (幻灯片{slide_num}, 图{img_idx}): {e}")

            if slide_texts:
                text_parts.append(f"[幻灯片 {slide_num}]\n" + "\n".join(slide_texts))

    except Exception as e:
        logger.error(f"PPT 解析失败: {e}")
        return FileParseResult(text=f"[错误：PPT 解析失败 - {e}]")

    return FileParseResult(text="\n\n".join(text_parts), images=images)
