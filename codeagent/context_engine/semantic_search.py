"""SemanticSearchEngine — 将代码向量化存储，支持自然语言检索。"""

from __future__ import annotations

import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable

import lancedb
import numpy as np

from codeagent.context_engine.code_chunker import CodeChunk, CodeChunker

# ── 默认配置 ───────────────────────────────────────────────────────────────────

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_DB_PATH = ".codeagent/lancedb"
_VECTOR_WEIGHT = 0.7
_BM25_WEIGHT = 0.3
_DEFAULT_TOP_K = 10


# ── SearchResult 数据类 ───────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """语义搜索结果。"""

    file_path: str
    start_line: int
    end_line: int
    code_snippet: str
    score: float
    symbol_name: str | None = None
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        d: dict[str, Any] = {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code_snippet": self.code_snippet,
            "score": round(self.score, 4),
        }
        if self.symbol_name is not None:
            d["symbol_name"] = self.symbol_name
        if self.language is not None:
            d["language"] = self.language
        return d


# ── 轻量 BM25 实现 ────────────────────────────────────────────────────────────


class _BM25Index:
    """轻量 BM25 检索，不需要额外依赖。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[str] = []
        self._doc_metadata: list[dict[str, Any]] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def _tokenize(self, text: str) -> list[str]:
        """将文本分词为小写 token。"""
        text = text.lower()
        # 保留代码 token：标识符、关键词、符号
        tokens = re.findall(r"[a-zA-Z_]\w*|[{}()\[\];,.:=+\-*/]", text)
        return tokens

    def index(self, docs: list[str], metadata: list[dict[str, Any]]) -> None:
        """索引文档列表。"""
        self._docs = docs
        self._doc_metadata = metadata

        # 计算文档频率
        N = len(docs)
        df: Counter = Counter()
        all_tokens: list[list[str]] = []
        total_terms = 0

        for doc in docs:
            tokens = self._tokenize(doc)
            all_tokens.append(tokens)
            unique_terms = set(tokens)
            df.update(unique_terms)
            total_terms += len(tokens)

        self._avgdl = total_terms / max(N, 1)

        # 计算 IDF
        for term, doc_freq in df.items():
            self._idf[term] = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

        self._all_tokens = all_tokens

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """对查询进行 BM25 检索，返回 (doc_index, score) 列表。"""
        if not self._docs:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: list[float] = []
        for i, tokens in enumerate(self._all_tokens):
            doc_len = len(tokens)
            score = 0.0
            for term in set(query_tokens):
                if term not in self._idf:
                    continue
                tf = tokens.count(term)
                idf = self._idf[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1))
                score += idf * numerator / denominator
            scores.append(score)

        # 排序并返回 top_k
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return [(idx, score) for idx, score in indexed_scores[:top_k] if score > 0]

    def get_metadata(self, idx: int) -> dict[str, Any]:
        """获取文档元数据。"""
        if 0 <= idx < len(self._doc_metadata):
            return self._doc_metadata[idx]
        return {}


# ── 嵌入模型封装 ──────────────────────────────────────────────────────────────


class _EmbeddingModel:
    """封装 sentence-transformers 嵌入模型。"""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None

    def _lazy_load(self) -> Any:
        """延迟加载模型。"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """将文本列表编码为向量。"""
        model = self._lazy_load()
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        """将查询编码为向量。"""
        model = self._lazy_load()
        # bge 模型建议为查询添加前缀
        embedding = model.encode(f"Represent this sentence for searching relevant passages: {query}", show_progress_bar=False)
        return embedding.tolist()


# ── 嵌入模型 Mock（用于测试） ────────────────────────────────────────────────


class _MockEmbeddingModel:
    """Mock 嵌入模型，用于测试。"""

    def __init__(self, model_name: str = "mock") -> None:
        self.model_name = model_name
        self._dim = 384

    def encode(self, texts: list[str]) -> list[list[float]]:
        """生成确定性 mock 向量。"""
        import hashlib

        result = []
        for text in texts:
            # 基于文本哈希生成确定性向量
            h = hashlib.md5(text.encode()).hexdigest()
            seed = int(h[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            result.append(vec.tolist())
        return result

    def encode_query(self, query: str) -> list[float]:
        """Mock 查询编码。"""
        return self.encode([query])[0]


# ── SemanticSearchEngine 主类 ─────────────────────────────────────────────────


class SemanticSearchEngine:
    """将代码向量化存储，支持自然语言检索。"""

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        model_name: str = _DEFAULT_MODEL,
        use_mock: bool = False,
    ) -> None:
        """初始化语义检索引擎。

        Args:
            db_path: LanceDB 数据库路径。
            model_name: 嵌入模型名称。
            use_mock: 是否使用 mock 嵌入模型（测试用）。
        """
        self._db_path = db_path
        self._model_name = model_name
        self._use_mock = use_mock

        # 延迟初始化
        self._model: _EmbeddingModel | _MockEmbeddingModel | None = None
        self._db: Any = None
        self._chunker = CodeChunker()
        self._bm25 = _BM25Index()

        # 索引统计
        self._total_chunks: int = 0
        self._total_files: int = 0
        self._chunk_metadata: list[dict[str, Any]] = []
        self._chunk_texts: list[str] = []

    def _get_model(self) -> _EmbeddingModel | _MockEmbeddingModel:
        """获取嵌入模型（延迟初始化）。"""
        if self._model is None:
            if self._use_mock:
                self._model = _MockEmbeddingModel(self._model_name)
            else:
                self._model = _EmbeddingModel(self._model_name)
        return self._model

    def _get_db(self) -> Any:
        """获取 LanceDB 数据库连接（延迟初始化）。"""
        if self._db is None:
            db_path = Path(self._db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(db_path))
        return self._db

    async def index_project(self, project_root: str) -> dict[str, Any]:
        """全量索引项目代码，返回索引统计。

        Args:
            project_root: 项目根目录路径。

        Returns:
            索引统计信息。
        """
        root = Path(project_root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        # 收集所有源文件
        extensions = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
        exclude_dirs = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            ".mypy_cache", ".pytest_cache", ".ruff_cache", ".egg-info",
            ".codeagent", ".claude", ".idea", ".vscode",
        }

        source_files: list[Path] = []
        try:
            for entry in root.rglob("*"):
                if entry.is_dir() and entry.name in exclude_dirs:
                    continue
                if entry.is_file() and entry.suffix.lower() in extensions:
                    source_files.append(entry)
        except (PermissionError, OSError):
            pass

        # 分块所有文件
        all_chunks: list[CodeChunk] = []
        for file_path in source_files:
            try:
                chunks = self._chunker.chunk_file(str(file_path))
                all_chunks.extend(chunks)
            except (FileNotFoundError, OSError, Exception):
                continue

        # 准备数据
        texts = [chunk.code for chunk in all_chunks]
        metadata = [chunk.to_dict() for chunk in all_chunks]

        if not texts:
            self._total_chunks = 0
            self._total_files = len(source_files)
            return {
                "total_files": self._total_files,
                "total_chunks": 0,
                "model": self._model_name,
            }

        # 生成嵌入向量
        model = self._get_model()
        embeddings = model.encode(texts)

        # 写入 LanceDB
        db = self._get_db()
        table_name = "code_chunks"

        try:
            db.drop_table(table_name)
        except Exception:
            pass

        table_data = []
        for i, chunk in enumerate(all_chunks):
            table_data.append({
                "vector": embeddings[i],
                "id": i,
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "code_snippet": chunk.code,
                "symbol_name": chunk.symbol_name or "",
                "language": chunk.language or "",
                "token_count": chunk.token_count,
            })

        table = db.create_table(table_name, data=table_data)

        # 创建 BM25 索引
        self._bm25.index(texts, metadata)

        # 更新统计
        self._total_chunks = len(all_chunks)
        self._total_files = len(source_files)
        self._chunk_metadata = metadata
        self._chunk_texts = texts

        return {
            "total_files": self._total_files,
            "total_chunks": self._total_chunks,
            "model": self._model_name,
        }

    async def search(
        self,
        query: str,
        top_k: int = _DEFAULT_TOP_K,
        filter_lang: str | None = None,
    ) -> list[SearchResult]:
        """混合检索：向量语义相似度（权重 0.7）+ BM25 关键词匹配（权重 0.3）。

        Args:
            query: 自然语言查询。
            top_k: 返回结果数量，默认 10。
            filter_lang: 按语言过滤（如 "python", "javascript"）。

        Returns:
            搜索结果列表，按混合得分降序排列。
        """
        if not self._total_chunks:
            return []

        db = self._get_db()
        table_name = "code_chunks"

        try:
            table = db.open_table(table_name)
        except Exception:
            return []

        # 向量搜索
        model = self._get_model()
        query_vector = model.encode_query(query)

        # 构建搜索
        search_builder = table.search(query_vector, vector_column_name="vector")

        # 语言过滤（lancedb 使用 SQL 风格的过滤语法）
        if filter_lang:
            search_builder = search_builder.where(f"language = '{filter_lang}'")

        # LanceDB 的距离是 L2，转换为相似度（1 / (1 + distance)）
        vector_results = search_builder.limit(top_k * 3).to_list()

        # 构建向量得分映射
        vector_scores: dict[int, float] = {}
        for r in vector_results:
            idx = r.get("id", -1)
            if idx >= 0:
                distance = r.get("_distance", 1.0)
                vector_scores[idx] = 1.0 / (1.0 + distance)

        # 归一化向量得分
        max_vec = max(vector_scores.values()) if vector_scores else 1.0
        if max_vec > 0:
            for idx in vector_scores:
                vector_scores[idx] /= max_vec

        # BM25 搜索（自行过滤语言）
        bm25_results = self._bm25.search(query, top_k=top_k * 3)
        bm25_scores: dict[int, float] = {}
        for idx, score in bm25_results:
            if filter_lang:
                meta = self._chunk_metadata[idx] if idx < len(self._chunk_metadata) else {}
                if meta.get("language") != filter_lang:
                    continue
            bm25_scores[idx] = score

        # 归一化 BM25 得分
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        if max_bm25 > 0:
            for idx in bm25_scores:
                bm25_scores[idx] /= max_bm25

        # 混合得分
        all_indices = set(vector_scores.keys()) | set(bm25_scores.keys())
        hybrid_scores: list[tuple[int, float]] = []

        for idx in all_indices:
            vec_score = vector_scores.get(idx, 0.0)
            bm25_score = bm25_scores.get(idx, 0.0)
            combined = _VECTOR_WEIGHT * vec_score + _BM25_WEIGHT * bm25_score
            hybrid_scores.append((idx, combined))

        # 排序取 top_k
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        hybrid_scores = hybrid_scores[:top_k]

        # 构建结果
        results: list[SearchResult] = []
        for idx, score in hybrid_scores:
            if idx < len(self._chunk_metadata):
                meta = self._chunk_metadata[idx]
                results.append(SearchResult(
                    file_path=meta.get("file_path", ""),
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    code_snippet=meta.get("code", ""),
                    score=score,
                    symbol_name=meta.get("symbol_name"),
                    language=meta.get("language"),
                ))

        return results

    async def reindex_file(self, file_path: str) -> None:
        """增量更新：仅重新索引变更的文件。

        Args:
            file_path: 变更文件的路径。
        """
        fp = Path(file_path)
        if not fp.exists():
            return

        # 重新分块该文件
        try:
            chunks = self._chunker.chunk_file(file_path)
        except (FileNotFoundError, OSError, Exception):
            return

        if not chunks:
            return

        db = self._get_db()
        table_name = "code_chunks"

        try:
            table = db.open_table(table_name)
        except Exception:
            return

        # 生成嵌入向量
        model = self._get_model()
        texts = [chunk.code for chunk in chunks]
        embeddings = model.encode(texts)

        # 删除该文件的旧索引
        try:
            table.delete(f'file_path = "{file_path.replace("\\", "\\\\")}"')
        except Exception:
            pass

        # 添加新索引
        table_data = []
        for i, chunk in enumerate(chunks):
            table_data.append({
                "vector": embeddings[i],
                "id": hash(chunk.code + str(chunk.start_line)),
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "code_snippet": chunk.code,
                "symbol_name": chunk.symbol_name or "",
                "language": chunk.language or "",
                "token_count": chunk.token_count,
            })

        if table_data:
            table.add(table_data)

        # 更新 BM25 和统计
        self._chunk_texts = []
        self._chunk_metadata = []

        # 重新加载所有数据
        try:
            data = table.to_pandas()
            if not data.empty:
                for _, row in data.iterrows():
                    self._chunk_texts.append(row.get("code_snippet", ""))
                    self._chunk_metadata.append({
                        "file_path": row.get("file_path"),
                        "start_line": row.get("start_line"),
                        "end_line": row.get("end_line"),
                        "code": row.get("code_snippet"),
                        "symbol_name": row.get("symbol_name"),
                        "language": row.get("language"),
                    })
                self._bm25.index(self._chunk_texts, self._chunk_metadata)

            self._total_chunks = len(data)
        except Exception:
            pass

    def get_index_stats(self) -> dict[str, Any]:
        """返回索引统计信息。"""
        return {
            "total_chunks": self._total_chunks,
            "total_files": self._total_files,
            "model": self._model_name,
            "db_path": self._db_path,
        }
