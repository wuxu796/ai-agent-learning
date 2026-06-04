"""
建库入口：读取提取好的章节 JSON → 切块 → 嵌入 → 存入 ChromaDB
用法: python build.py
"""
import json
import time
from pathlib import Path
from engine import BookRAG


def main():
    # 1. 读取提取结果
    input_file = Path("extracted/chapters_full.json")
    if not input_file.exists():
        print(f"❌ 找不到 {input_file}")
        print("   请先运行: python extract.py")
        return

    print(f"正在加载 {input_file} ...")
    with open(input_file, "r", encoding="utf-8") as f:
        chapters = json.load(f)

    print(f"已加载 {len(chapters)} 个章节，总 {sum(c['char_count'] for c in chapters):,} 字")

    # 2. 确认
    print(f"\n配置:")
    print(f"  模型: paraphrase-multilingual-MiniLM-L12-v2")
    print(f"  切块: 400字/块, 80字重叠")
    print(f"  嵌入批次: 每批500块")
    print(f"  预计块数: ~{sum(c['char_count'] for c in chapters) // 320:,}")
    print(f"  预计耗时: 1-3分钟")
    print()

    choice = input("开始建库？(y/n): ").strip().lower()
    if choice != "y":
        print("已取消。")
        return

    # 3. 建库
    engine = BookRAG(collection_name="nutrition_science")

    start = time.time()
    total = engine.build_from_chapters(
        chapters,
        chunk_size=400,
        overlap=80,
        batch_size=500,
    )
    elapsed = time.time() - start

    # 4. 结果
    print(f"\n{'='*50}")
    print(f"  建库完成！")
    print(f"  总块数: {total:,}")
    print(f"  耗时: {elapsed:.0f}秒 ({elapsed/60:.1f}分钟)")
    print(f"  存储位置: ./chroma_db/")
    print(f"\n  下一步: python app.py")


if __name__ == "__main__":
    main()
