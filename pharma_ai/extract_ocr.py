"""
药典 PDF 提取 — PaddleOCR 版
============================
在 VM 上运行（Python 3.10 + PaddlePaddle）
用法：
  python3 extract_ocr.py --part1-only --limit=5    # 一部前5页测试
  python3 extract_ocr.py --part1-only              # 一部全量
  python3 extract_ocr.py                           # 一部+二部全量
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np
from paddleocr import PaddleOCR

# ==================== 配置 ====================

BOOKS = [
    {
        "file": "/home/wuxu/pharma_books/《中国药典》2025年版 一部 全本.pdf",
        "name": "一部（中药）",
        "start_page": 51,
        "dpi": 300,
    },
    {
        "file": "/home/wuxu/pharma_books/《中国药典》2025年版 二部.pdf",
        "name": "二部（化学药）",
        "start_page": 51,
        "dpi": 300,
    },
]

OUTPUT_DIR = Path("/home/wuxu/pharma_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==================== 初始化 PaddleOCR ====================

print("初始化 PaddleOCR ...")
ocr = PaddleOCR(lang="ch")
print("PaddleOCR 就绪\n")


# ==================== 清洗 ====================


def clean_ocr(text: str) -> str:
    """过滤页眉、修复标点"""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if re.match(r"^[画中]?国药典\s*\d{4}\s*年版", line):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)

    text = text.replace("〗", "】").replace("〖", "【")
    text = re.sub(r"【\s*", "【", text)
    text = re.sub(r"\s*】", "】", text)
    text = re.sub(r"（\s*", "（", text)
    text = re.sub(r"\s*）", "）", text)
    text = re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", text)
    return text


# ==================== OCR 单页 ====================


def ocr_page(pdf_doc, page_idx: int, dpi: int = 300) -> str:
    """PaddleOCR 单页，裁顶 10% 去页眉"""
    page = pdf_doc[page_idx]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    # 裁掉顶部 10%
    crop_top = int(img.height * 0.10)
    img = img.crop((0, crop_top, img.width, img.height))

    # PaddleOCR 支持 PIL Image 或 numpy array
    img_array = np.array(img)
    result = ocr.ocr(img_array)

    if not result or not result[0]:
        return ""

    # PaddleOCR 输出格式: [[[bbox], (text, confidence)], ...]
    lines = []
    for line in result[0]:
        text = line[1][0]  # 取文字
        lines.append(text)

    text = "\n".join(lines)
    return clean_ocr(text)


# ==================== 主流程 ====================


def extract_book(book_config: dict, limit: int | None = None):
    filepath = book_config["file"]
    name = book_config["name"]
    start_page = book_config["start_page"]
    dpi = book_config["dpi"]

    if not os.path.exists(filepath):
        print(f"[X] 文件不存在: {filepath}")
        return []

    print(f"\n{'='*50}")
    print(f"[开始] {name}")
    print(f"   起始页: {start_page}  DPI: {dpi}")
    print(f"{'='*50}")

    doc = fitz.open(filepath)
    total_pages = doc.page_count
    pages_to_do = list(range(start_page - 1, total_pages))

    if limit:
        pages_to_do = pages_to_do[:limit]
        print(f"   [限制] 只处理前 {limit} 页")

    pages_output = []
    start_time = time.time()

    for i, pg_idx in enumerate(pages_to_do):
        try:
            text = ocr_page(doc, pg_idx, dpi)
            pages_output.append({
                "page": pg_idx + 1,
                "file": name,
                "text": text,
            })

            if (i + 1) % 10 == 0 or i == len(pages_to_do) - 1:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                eta = (len(pages_to_do) - i - 1) / rate if rate > 0 else 0
                print(f"  进度: {i+1}/{len(pages_to_do)}  ({rate:.1f}页/分  剩余{eta:.0f}分)")

        except Exception as e:
            print(f"   [WARN] 第 {pg_idx+1} 页出错: {e}")

    doc.close()

    elapsed = time.time() - start_time
    print(f"\n   [OK] {name} 完成！{len(pages_to_do)} 页 / {elapsed:.1f}分")
    return pages_output


def main():
    if "--part1-only" in sys.argv:
        books = [BOOKS[0]]
    elif "--part2-only" in sys.argv:
        books = [BOOKS[1]]
    else:
        books = BOOKS

    limit = None
    for arg in sys.argv:
        if arg.startswith("--limit="):
            limit = int(arg.split("=")[1])

    all_pages = []
    for book in books:
        pages = extract_book(book, limit=limit)
        all_pages.extend(pages)

    output_json = OUTPUT_DIR / "pharmacopoeia_raw.json"
    output_txt = OUTPUT_DIR / "pharmacopoeia_raw.txt"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)
    with open(output_txt, "w", encoding="utf-8") as f:
        for p in all_pages:
            f.write(f"=== 第{p['page']}页 ({p['file']}) ===\n")
            f.write(p["text"])
            f.write("\n\n")

    total_chars = sum(len(p["text"]) for p in all_pages)
    print(f"\n{'='*50}")
    print(f"[Done] 输出: {output_json}")
    print(f"       {output_txt}")
    print(f"   总页数: {len(all_pages)}  总字数: {total_chars}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
