"""
Redis 缓存模块 — 给食鉴贴便利贴

类比：
  你查营养全书（翻5635页 = 搜 ChromaDB 向量库）→ 慢
  有了 Redis，就像把查过的答案写在便利贴贴桌上：
    第二次问同样的问题 → 直接从便利贴拿 → 快 100 倍

两个缓存场景：
  1. 搜索缓存 — 用 MD5(问题) 做 key，存向量库返回结果，1小时后自动撕掉
  2. 会话缓存 — 用 session_id 做 key，存整个对话历史，7天后自动撕掉

连接方式：
  不同环境自动适配 Redis 地址：
  - 宿主机直接跑 → localhost:6379
  - Docker 容器里跑 → 看 REDIS_HOST 环境变量
"""
import hashlib
import json
import os
from typing import Optional

import redis


class Cache:
    """Redis 缓存封装，所有操作自动处理连接失败（降级为无缓存）"""

    def __init__(self, host: str = "localhost", port: int = 6379):
        self.host = host
        self.port = port
        self._r: Optional[redis.Redis] = None

    @property
    def r(self) -> Optional[redis.Redis]:
        """延迟连接：第一次用时才连，避免 import 时就报错"""
        if self._r is None:
            try:
                self._r = redis.Redis(
                    host=self.host,
                    port=self.port,
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                self._r.ping()  # 测试通不通
                print(f"[缓存] Redis 已连接 ({self.host}:{self.port})")
            except Exception as e:
                print(f"[缓存] Redis 不可用 ({e})，缓存功能关闭，服务器照常运行")
                self._r = False  # 标记为已尝试过
        return self._r if self._r is not False else None

    # ==================== 搜索缓存 ====================

    def _search_key(self, query: str) -> str:
        """把问题变成 Redis key：shijian:search:md5哈希值"""
        md5 = hashlib.md5(query.encode("utf-8")).hexdigest()
        return f"shijian:search:{md5}"

    def get_search(self, query: str) -> Optional[str]:
        """查便利贴：有没有这个问题缓存过的结果？"""
        r = self.r
        if r is None:
            return None
        try:
            return r.get(self._search_key(query))
        except Exception:
            return None

    def set_search(self, query: str, result: str, ttl: int = 3600):
        """贴便利贴：把搜索结果存起来。默认 1 小时后自动撕掉（TTL）"""
        r = self.r
        if r is None:
            return
        try:
            r.setex(self._search_key(query), ttl, result)
        except Exception:
            pass

    # ==================== 会话缓存 ====================

    def _session_key(self, session_id: str) -> str:
        return f"shijian:session:{session_id}"

    def get_session(self, session_id: str) -> Optional[list]:
        """从 Redis 读会话（内存→秒读）"""
        r = self.r
        if r is None:
            return None
        try:
            data = r.get(self._session_key(session_id))
            return json.loads(data) if data else None
        except Exception:
            return None

    def set_session(self, session_id: str, messages: list, ttl: int = 604800):
        """
        把会话存 Redis。
        TTL 默认 7 天（604800秒），过期自动删除，不会撑爆内存。
        """
        r = self.r
        if r is None:
            return
        try:
            r.setex(
                self._session_key(session_id),
                ttl,
                json.dumps(messages, ensure_ascii=False),
            )
        except Exception:
            pass

    def delete_session(self, session_id: str):
        """删除 Redis 中的会话"""
        r = self.r
        if r is None:
            return
        try:
            r.delete(self._session_key(session_id))
        except Exception:
            pass


# ==================== 全局单例 ====================

# 容器内用环境变量 REDIS_HOST，宿主机用 localhost
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
cache = Cache(host=REDIS_HOST)
