# 食鉴

基于《中国营养科学全书（第2版）》的 RAG 智能问答系统。505 万字全文索引，自然语言检索。

## 架构

```
浏览器 (HTML/JS) ←→ FastAPI ←→ ChromaDB (13,933 块)
     ↑                  ↓
     │           LLM 工具编排
     │           ├─ calculate_bmi
     │           ├─ calculate_tdee
     │           └─ search_book
     │                  ↓
     │        DeepSeek API / Ollama 本地
     │        (LLM_PROVIDER 环境变量切换)
     │                  ↓
     └──── 保底策略（本地模型漏调工具时自动补算 + 搜书）
```

## 技术栈

| 层 | 技术 | 说明 |
|---|------|------|
| PDF 提取 | PyMuPDF | 章节感知提取，目录层级重建，1927 章 |
| 嵌入 | sentence-transformers (MiniLM-L12-v2) | 384 维中文向量 |
| 向量库 | ChromaDB | 本地持久化，L2 归一化 |
| 后端 | FastAPI | REST API + 多会话 JSON 持久化 |
| 前端 | HTML/CSS/JS（零框架） | Markdown 渲染、复制按钮、可折叠侧边栏 |
| LLM | DeepSeek API / Ollama | 一行环境变量切换 |
| 缓存 | Redis | 搜索缓存 + 会话缓存，TTL 自动过期 |
| 部署 | Docker Compose | 一键编排，食鉴 + Redis 一起管 |

## 快速开始

### Docker Compose（推荐）

```bash
# 1. 创建 .env
echo 'DEEPSEEK_API_KEY=你的key' > .env
echo 'LLM_PROVIDER=deepseek' >> .env

# 2. 一键启动（食鉴 + Redis）
docker-compose up -d --build
```

浏览器打开 http://localhost:8000

> 首次构建约 5-10 分钟。后续启动秒级。
> 知识库需提前构建（见下方"构建知识库"），或从已有环境拷贝 `chroma_db/` 目录。

### 手动安装

### 1. 安装依赖

```bash
pip install fastapi uvicorn openai sentence-transformers chromadb pymupdf python-dotenv
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key
```

### 3. 构建知识库（仅首次）

```bash
python build.py
```

预计 1-3 分钟，消耗 1-2GB 显存，完成后 `chroma_db/` 约 25MB。

### 4. 启动

```bash
python server.py
```

浏览器打开 http://127.0.0.1:8000

## 本地模型（可选）

RTX 4060 Ti 8GB 实测可用。无需 API Key，数据不出本机。

```bash
# 安装 Ollama → https://ollama.com
ollama pull qwen2.5:7b

# 修改 .env
LLM_PROVIDER=ollama
```

不使用时释放显存：

```bash
ollama stop qwen2.5:7b
```

> **7B 模型的局限**：Function Calling 不如 DeepSeek 稳定，可能漏调工具或直接编造数字。本项目已内置保底策略——后端自动从消息中提取身高/体重计算 BMI，自动搜索知识库，大幅降低幻觉。但极端情况下仍可能出现偏差，重要查询建议切回 DeepSeek。

## 项目结构

```
nutri_ai/
├── docker-compose.yml    # 食鉴 + Redis 一键编排
├── Dockerfile
├── cache.py              # Redis 缓存：搜索 + 会话
├── server.py             # FastAPI 后端
├── engine.py             # RAG 引擎
├── extract.py            # PDF 提取
├── build.py              # 知识库构建
├── static/
│   └── index.html
├── extracted/
├── chroma_db/
├── sessions/
├── .env.example
└── README.md
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/chat` | 发送消息 `{"session_id":"...", "message":"..."}` |
| `POST` | `/api/sessions` | 创建新会话 |
| `GET` | `/api/sessions` | 列出所有会话 |
| `GET` | `/api/sessions/{id}` | 获取会话聊天记录 |
| `DELETE` | `/api/sessions/{id}` | 删除会话 |
| `GET` | `/api/health` | 健康检查（模型、知识库状态） |

## 工具编排

Agent 自动判断调用哪个工具：

| 工具 | 触发条件 | 实现 |
|------|---------|------|
| `calculate_bmi` | 用户提供身高 + 体重 | Python 精确计算 |
| `calculate_tdee` | 用户提供性别/年龄/身高/体重/活动量 | Mifflin-St Jeor 公式 |
| `search_book` | 营养学知识问题 | 向量检索 → ChromaDB top-5 |

本地模型漏调工具时，后端自动触发补算和搜书，确保回答有据可依。

## 从 V1 到 V2

|     | V1（小行健康助手）     | V2（食鉴）                 |
| --- | -------------- | ---------------------- |
| 知识库 | 7 篇手写文章        | 《中国营养科学全书》（第二版）        |
| 切块  | 简单按标点          | 章节边界约束 + 元数据           |
| 嵌入  | 一次性全量          | 分批嵌入，防 OOM             |
| 前端  | Gradio 组件      | 自写 HTML，多会话 + Markdown |
| 模型  | 仅 DeepSeek API | DeepSeek / Ollama 双后端  |
| 健壮性 | 依赖 LLM 判断      | + 保底策略自动补算搜书           |
