"""
PDF 提取：章节感知抽取全文
输出 extracted/chapters.json — 每章的结构化文本 + 元数据
"""
import fitz
import json
import os
import re
from pathlib import Path

PDF_PATH = input("PDF路径: ").strip().strip('"')

# 输出目录
OUT_DIR = Path("extracted")
OUT_DIR.mkdir(exist_ok=True)

print("正在打开 PDF（5635页大文件，稍等）...")
doc = fitz.open(PDF_PATH)
total_pages = doc.page_count
print(f"总页数: {total_pages}")

# ==================== 1. 解析目录树 ====================
print("\n[1/4] 解析章节目录...")
raw_toc = doc.get_toc()

if not raw_toc:
    print("❌ PDF 没有内嵌目录，无法按章节切分。请换一个带书签的 PDF。")
    doc.close()
    exit(1)

print(f"  目录条目数: {len(raw_toc)}")

# TOC 格式: [(level, title, page), ...]
# 构建层级结构
chapters = []  # [{title, page_start, page_end, level, path}]
prev = None
for i, (level, title, page) in enumerate(raw_toc):
    title = title.strip()
    if prev:
        prev["page_end"] = page - 1  # 上一章的结束页
    chapters.append({
        "level": level,         # 1=卷, 2=章, 3=节...
        "title": title,
        "page_start": page - 1,  # 转为0-based
        "page_end": None,
        "path": [],              # 面包屑，如 ["第一卷", "第一章", "蛋白质"]
    })
    prev = chapters[-1]

# 最后一章到末尾
if chapters:
    chapters[-1]["page_end"] = total_pages - 1

# 构建面包屑路径
stack = [None]  # level 0 占位
for ch in chapters:
    lvl = ch["level"]
    # 弹出比当前层级深的
    while len(stack) > lvl:
        stack.pop()
    # 当前层级的标题
    ch["path"] = [s["title"] for s in stack[1:]] + [ch["title"]]
    stack.append(ch)

print(f"  卷/章/节 共 {len(chapters)} 个节点")
print(f"  最大层级: {max(c['level'] for c in chapters)}")

# 打印前几层看看结构
for ch in chapters[:10]:
    indent = "  " * (ch["level"] - 1)
    pg_range = f"p.{ch['page_start']+1}-{ch['page_end']+1}"
    print(f"  {indent}[{pg_range}] {ch['title']}")

# ==================== 2. 按章节提取文字 ====================
print(f"\n[2/4] 按章节提取文字...")

# 找出目录的有效章节（level 1-2 太大，level>=3 太碎，主要用 level 1-2 的章节）
# level 1 = 卷（7卷），每卷下有 level 2 = 章
# 用 level 2 作为存储单元

# 实际上，用户搜索时我们想要中粒度。用最小章节（leaf-level）作为chunk的元数据边界。
# 这里先把所有文本按章节提取出来，存为中间文件。

chapter_texts = []  # [{title, path, page_start, page_end, text}]

for idx, ch in enumerate(chapters):
    # 只处理有实际页数范围的（跳过太短的目录条目）
    pg_count = ch["page_end"] - ch["page_start"] + 1
    if pg_count < 1:
        continue

    if (idx + 1) % 50 == 0:
        print(f"  进度: {idx+1}/{len(chapters)} 章节...")

    text_parts = []
    for pg in range(ch["page_start"], ch["page_end"] + 1):
        try:
            page_text = doc[pg].get_text()
            if page_text.strip():
                text_parts.append(page_text)
        except Exception:
            continue

    full_text = "\n".join(text_parts)
    if len(full_text.strip()) < 50:  # 跳过几乎空白的章节
        continue

    chapter_texts.append({
        "title": ch["title"],
        "path": ch["path"],
        "level": ch["level"],
        "page_start": ch["page_start"],
        "page_end": ch["page_end"],
        "char_count": len(full_text),
        "text": full_text,
    })

print(f"  有效章节数: {len(chapter_texts)}")
total_chars = sum(c["char_count"] for c in chapter_texts)
print(f"  总字数: {total_chars:,}（{total_chars/10000:.1f}万字）")

# ==================== 3. 清洗文本 ====================
print(f"\n[3/4] 清洗文本...")

def clean_text(text):
    """去除PDF提取常见噪声"""
    # 合并被切断的行（中文字段末尾没有标点，下一行是中文开头）
    text = re.sub(r'(?<=[^。！？；\.\n])\n(?=[^\n]{2,})', '', text)
    # 多余空行压缩
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 页码数字（单独一行的数字）
    text = re.sub(r'^\d{1,4}$', '', text, flags=re.MULTILINE)
    # 页眉页脚常见模式
    text = re.sub(r'中国营养科学全书.*?\n', '', text)
    return text.strip()

for ch in chapter_texts:
    ch["text"] = clean_text(ch["text"])
    ch["char_count"] = len(ch["text"])

# ==================== 4. 保存 ====================
print(f"\n[4/4] 保存到 {OUT_DIR}/...")

# 全量 JSON（可能几百MB）
full_path = OUT_DIR / "chapters_full.json"
with open(full_path, "w", encoding="utf-8") as f:
    json.dump(chapter_texts, f, ensure_ascii=False)
print(f"  ✓ chapters_full.json ({full_path.stat().st_size / 1024 / 1024:.1f} MB)")

# 摘要（不含文本，方便查看结构）
summary = [{k: v for k, v in ch.items() if k != "text"} for ch in chapter_texts]
with open(OUT_DIR / "chapters_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"  ✓ chapters_summary.json")

# 统计信息
stats = {
    "pdf_path": PDF_PATH,
    "total_pages": total_pages,
    "toc_entries": len(raw_toc),
    "valid_chapters": len(chapter_texts),
    "total_chars": total_chars,
    "total_chars_wan": round(total_chars / 10000, 1),
    "max_depth": max(c["level"] for c in chapters),
    "volumes": [c for c in chapter_texts if c["level"] == 1],
}
with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(f"  ✓ stats.json")

doc.close()
print(f"\n✅ 提取完成！下一步: python build.py")
