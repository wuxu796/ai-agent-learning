"""
RAG 引擎 V2 — 全量书籍版
改进：
  1. 分批嵌入（不爆显存）
  2. 章节感知切块（不跨章节边界）
  3. 元数据存储（每块带卷/章/节/页码）
  4. 来源引用检索（回答可溯源）
"""
import re
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class BookRAG:
    """面向大篇幅书籍的 RAG 引擎"""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        chroma_path: str = "./chroma_db",
        collection_name: str = "nutrition_science",
    ):
        print(f"[引擎] 加载嵌入模型: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        print("[引擎] 模型加载完成。")

        self.chroma_path = chroma_path
        self.collection_name = collection_name
        os.makedirs(chroma_path, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(collection_name)

    # ==================== 切块（章节感知） ====================

    def chunk_chapter(
        self,
        chapter: dict,
        chunk_size: int = 400,
        overlap: int = 80,
    ) -> list[dict]:
        """
        对单个章节切块，每个块自动带上章节元数据。
        chapter: {"title", "path", "page_start", "page_end", "text", ...}
        返回: [{"text", "source_title", "source_path", "page_range", "chunk_index"}, ...]
        """
        text = chapter["text"].strip()
        if not text:
            return []

        # 按句子边界切
        sentences = re.split(r"(?<=[。！？；\n])", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current = ""
        chunk_idx = 0

        for sent in sentences:
            if len(current) + len(sent) > chunk_size and len(current) >= 60:
                chunks.append({
                    "text": current,
                    "source_title": chapter["title"],
                    "source_path": " > ".join(chapter["path"]),
                    "page_range": f"第{chapter['page_start']+1}-{chapter['page_end']+1}页",
                    "chunk_index": chunk_idx,
                })
                chunk_idx += 1
                carry = current[-overlap:] if len(current) > overlap else current
                current = carry + sent
            else:
                current += sent

        if len(current) >= 60:
            chunks.append({
                "text": current,
                "source_title": chapter["title"],
                "source_path": " > ".join(chapter["path"]),
                "page_range": f"第{chapter['page_start']+1}-{chapter['page_end']+1}页",
                "chunk_index": chunk_idx,
            })

        return chunks

    # ==================== 建库（分批写入） ====================

    def build_from_chapters(
        self,
        chapters: list[dict],
        chunk_size: int = 400,
        overlap: int = 80,
        batch_size: int = 500,
    ) -> int:
        """
        从章节列表构建知识库。
        流程：逐章切块 → 分批嵌入 → 分批写入 ChromaDB
        batch_size: 每批嵌入的块数（4090/4060 Ti 建议 500-800）
        """
        print(f"\n[建库] 共 {len(chapters)} 个章节，正在切块...")

        # 第一步：全量切块（切块本身不占显存，很快）
        all_chunks = []
        for i, ch in enumerate(chapters):
            ch_chunks = self.chunk_chapter(ch, chunk_size, overlap)
            all_chunks.extend(ch_chunks)
            if (i + 1) % 100 == 0:
                print(f"  切块进度: {i+1}/{len(chapters)} 章节, 已累积 {len(all_chunks)} 块")

        if not all_chunks:
            print("[建库] 警告：没有切出任何文本块。")
            return 0

        print(f"[建库] 切块完成，共 {len(all_chunks)} 块。")

        # 第二步：清空旧库
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(self.collection_name)

        # 第三步：分批嵌入 + 写入
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        print(f"[建库] 开始分批嵌入，共 {total_batches} 批（每批 {batch_size} 块）...")

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(all_chunks))
            batch_chunks = all_chunks[start:end]

            texts = [c["text"] for c in batch_chunks]

            # 编码 + L2归一化
            embeddings = self.model.encode(texts, show_progress_bar=False)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # 准备元数据
            ids = [f"chunk_{i}" for i in range(start, end)]
            metadatas = [
                {
                    "source_title": c["source_title"],
                    "source_path": c["source_path"],
                    "page_range": c["page_range"],
                }
                for c in batch_chunks
            ]

            # 写入
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )

            progress = (batch_idx + 1) / total_batches * 100
            print(f"  [{batch_idx+1}/{total_batches}] {end}/{len(all_chunks)} 块 ({progress:.0f}%)")

        print(f"[建库] ✅ 完成！共存储 {len(all_chunks)} 个文本块。")
        return len(all_chunks)

    # ==================== 检索 ====================

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """
        检索并返回带来源的结果列表。
        返回: [{"text", "title", "path", "page", "relevance"}, ...]
        """
        if self.collection.count() == 0:
            return []

        # 编码查询
        query_emb = self.model.encode(query)
        query_emb = query_emb / np.linalg.norm(query_emb)
        query_emb = query_emb.tolist()

        result = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        results = []
        for doc, meta, dist in zip(docs, metas, dists):
            dist_val = float(dist) if dist is not None else 999
            relevance = max(0, round(1 - dist_val / 10, 2))
            results.append({
                "text": doc,
                "title": meta.get("source_title", "未知") if meta else "未知",
                "path": meta.get("source_path", "") if meta else "",
                "page": meta.get("page_range", "") if meta else "",
                "relevance": relevance,
            })

        return results

    def search_and_format(self, query: str, n_results: int = 5) -> str:
        """检索并格式化为 LLM 可用的上下文。"""
        results = self.search(query, n_results)

        if not results:
            return "知识库中没有找到相关信息。"

        lines = []
        for r in results:
            lines.append(r["text"])
            lines.append("")

        return "\n".join(lines)

    # ==================== 工具方法 ====================

    def get_chunk_count(self) -> int:
        return self.collection.count()

    def get_stats(self) -> dict:
        """返回知识库统计信息"""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "chroma_path": self.chroma_path,
        }
