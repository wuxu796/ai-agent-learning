"""
食鉴 — LangChain 后端（与 server.py 并列，不修改原文件）

对比计划：
  server.py          → 手写工具编排（if-elif 路由），端口 8000
  server_langchain.py → LangChain Agent 编排，端口 8001

同一个前端、同一套工具，只换后端引擎。
"""
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# ==================== LLM ====================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()

if LLM_PROVIDER == "ollama":
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    MODEL = OLLAMA_MODEL
    print(f"[LLM] 本地模型: {MODEL} @ {OLLAMA_URL}")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=MODEL, base_url=OLLAMA_URL, api_key="ollama", temperature=0)
else:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("请设置 DEEPSEEK_API_KEY 或 LLM_PROVIDER=ollama")
    MODEL = "deepseek-chat"
    print(f"[LLM] DeepSeek API")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0,
    )

# ==================== RAG 引擎（复用 engine.py） ====================

from engine import BookRAG
rag = BookRAG()

# ==================== 工具（@tool 替代手写 JSON schema） ====================

from langchain.tools import tool

ACTIVITY_LEVELS = {
    "久坐": 1.2, "轻度": 1.375, "中度": 1.55, "高度": 1.725, "运动员": 1.9,
}


@tool
def calculate_bmi(weight: float, height: float) -> str:
    """计算身体质量指数 BMI。用户提供体重（公斤）和身高（厘米）时调用。"""
    if height <= 0 or weight <= 0:
        return "身高和体重必须大于 0。"
    height_m = height / 100
    bmi = round(weight / (height_m ** 2), 1)
    if bmi < 18.5:
        level, tip = "偏瘦", "可以适当增重，多吃蛋白质和优质碳水。"
    elif bmi < 24:
        level, tip = "正常", "体重在健康范围，继续保持！"
    elif bmi < 28:
        level, tip = "偏胖", "建议控制饮食 + 增加有氧运动。"
    else:
        level, tip = "肥胖", "建议制定系统的减脂计划，注意饮食和运动。"
    return f"BMI = {bmi}（{level}）。{tip}"


@tool
def calculate_tdee(gender: str, age: int, weight: float, height: float, activity: str) -> str:
    """计算每日总能量消耗 TDEE。用户想知道每天消耗多少热量、减脂吃多少时调用。
    gender: '男' 或 '女'。activity: 久坐/轻度/中度/高度/运动员。"""
    if age <= 0 or weight <= 0 or height <= 0:
        return "年龄、体重、身高必须大于 0。"
    if gender == "男":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    elif gender == "女":
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        return "性别请输入'男'或'女'。"
    ratio = ACTIVITY_LEVELS.get(activity, 1.2)
    tdee = round(bmr * ratio)
    return (
        f"基础代谢（BMR）：{round(bmr)} 千卡/天\n"
        f"每日消耗（TDEE）：{tdee} 千卡/天（{activity}模式，系数{ratio}）\n"
        f"减脂建议：每天摄入约 {tdee - 500} 千卡\n"
        f"增肌建议：每天摄入约 {tdee + 300} 千卡"
    )


@tool
def search_book(query: str) -> str:
    """搜索《中国营养科学全书》知识库。用户问到营养学理论、营养素功能、代谢机制、
    膳食指南、食物成分、各类人群营养需求等专业知识时，务必调用此工具。"""
    if rag.get_chunk_count() == 0:
        return "知识库为空，请先运行 python build.py 构建知识库。"
    return rag.search_and_format(query, n_results=5)


tools = [calculate_bmi, calculate_tdee, search_book]

# ==================== LangChain Agent ====================

from langchain.agents import initialize_agent, AgentType

# 系统提示（和手写版 SYSTEM_PROMPT 保持一致）
SYSTEM_PROMPT = (
    "你叫食鉴，是一个营养健康助手。\n"
    "你可以帮用户：\n"
    "1. 计算 BMI（身体质量指数）→ 必须调用 calculate_bmi 工具，不要自己算！\n"
    "2. 计算 TDEE（每日消耗热量）→ 必须调用 calculate_tdee 工具，不要自己算！\n"
    "3. 回答营养学问题 → 必须调用 search_book 工具搜书，不要凭记忆回答！\n\n"
    "回答规则：\n"
    "- 风格简洁、专业，用生活化的语言解释专业概念。\n"
    "- 禁止在回复中提及任何信息来源，如\"根据全书\"、\"书中记载\"等。\n"
    "- 回复中避免使用 ~ 符号，用\"到\"字代替。\n"
    "- 给出具体数字（BMI、TDEE、热量等）之前，必须先调用工具获取准确结果。"
)

# 每个会话独立一个 agent（带记忆）
from langchain.memory import ConversationBufferMemory

agents: dict[str, any] = {}


def get_agent(session_id: str):
    """为每个会话创建独立的 Agent（绑定了独立记忆）"""
    if session_id not in agents:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=False,
            handle_parsing_errors=True,
            memory=memory,
            agent_kwargs={"system_message": SYSTEM_PROMPT},
        )
        agents[session_id] = agent
    return agents[session_id]


# ==================== 保底策略（复用 server.py 的逻辑） ====================

from cache import cache


def _fallback_check(user_message: str, agent_reply: str) -> str | None:
    """
    本地模型保底：如果 agent 没调工具，后端自动补算 BMI/TDEE + 搜书。
    返回 None 表示不需要补算，返回字符串表示补算结果。
    """
    if LLM_PROVIDER != "ollama":
        return None

    # 检查 agent 回复里有没有数字（如果有，说明可能自己编了）
    has_numbers = bool(re.search(r"\d+", agent_reply))

    fallback_parts = []

    h_match = re.search(r"身高\s*(\d{2,3})", user_message)
    w_match = re.search(r"体重\s*(\d{2,3})", user_message)

    if h_match and w_match:
        height = int(h_match.group(1))
        weight = int(w_match.group(1))
        fallback_parts.append(f"[BMI自动计算]\n{calculate_bmi(weight, height)}")

        g_match = re.search(r"(?:性别[：:]*)?(男|女)(?:[生性]|\b)", user_message)
        a_match = re.search(r"(?:年龄[：:]*)?(\d{1,3})\s*岁", user_message)
        act_match = re.search(
            r"(久坐|轻度|中度|高度|运动员|不怎么动|很少运动|每周运动|经常运动|办公室)",
            user_message,
        )
        if g_match and a_match and act_match:
            gender = g_match.group(1)
            age = int(a_match.group(1))
            raw_act = act_match.group(1)
            act_map = {
                "不怎么动": "久坐", "很少运动": "久坐", "办公室": "久坐",
                "每周运动": "中度", "经常运动": "高度",
            }
            activity = act_map.get(raw_act, raw_act)
            if activity in ACTIVITY_LEVELS:
                fallback_parts.append(
                    f"[TDEE自动计算]\n{calculate_tdee(gender, age, weight, height, activity)}"
                )

    clean_query = re.sub(r"身高\s*\d{2,3}", "", user_message)
    clean_query = re.sub(r"体重\s*\d{2,3}", "", clean_query)
    clean_query = re.sub(r"\d{1,3}\s*岁", "", clean_query)
    clean_query = re.sub(r"[男女]生?", "", clean_query)
    book_result = search_book(clean_query.strip() or user_message)
    if book_result and "没有找到相关信息" not in book_result:
        fallback_parts.append(f"[知识库检索]\n{book_result}")

    if fallback_parts:
        return "\n\n".join(fallback_parts)
    return None


# ==================== 会话持久化（JSON 文件 + Redis 双写） ====================

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def load_session(session_id: str) -> list[dict]:
    data = cache.get_session(session_id)
    if data is not None:
        return data
    path = _session_path(session_id)
    if path.exists():
        messages = json.loads(path.read_text(encoding="utf-8"))
        cache.set_session(session_id, messages)
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def save_session(session_id: str, messages: list[dict]):
    _session_path(session_id).write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cache.set_session(session_id, messages)


# ==================== FastAPI ====================

PORT = int(os.getenv("PORT", "8001"))   # 默认 8001，不和原版冲突
app = FastAPI(title="食鉴-LangChain版")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


def _clean_response(text: str) -> str:
    text = re.sub(r"<tool_calls>[^<]*</tool_calls>", "", text, flags=re.DOTALL)
    text = re.sub(r"<invoke[^>]*>[\s\S]*?</invoke>", "", text)
    text = re.sub(r"</?(?:tool_calls|invoke|function|parameter)[^>]*>", "", text)
    return text.strip()


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "消息不能为空")

    agent = get_agent(req.session_id)

    # LangChain 的 run() 自动完成：工具决策 → 调用 → 结果拼接 → LLM 合成
    reply = agent.run(req.message)
    reply = _clean_response(reply)

    # 保底策略（本地模型）
    fallback = _fallback_check(req.message, reply)
    if fallback:
        fallback_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=False,
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": (
                    "以下数据由系统自动计算和检索，你必须使用这些数据来回答，不要自己编造数字。\n"
                    + fallback
                ),
            },
        )
        reply = fallback_agent.run(req.message)
        reply = _clean_response(reply)

    # 持久化到 JSON + Redis
    messages = load_session(req.session_id)
    messages.append({"role": "user", "content": req.message})
    messages.append({"role": "assistant", "content": reply})
    save_session(req.session_id, messages)

    return ChatResponse(session_id=req.session_id, reply=reply)


@app.get("/api/sessions")
def list_sessions():
    sessions = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            messages = json.loads(path.read_text(encoding="utf-8"))
            preview = ""
            for m in messages:
                if m["role"] == "user":
                    preview = m["content"][:40]
                    break
            sessions.append({
                "session_id": path.stem,
                "created_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "message_count": sum(1 for m in messages if m["role"] in ("user", "assistant")),
                "preview": preview or "(空会话)",
            })
        except Exception:
            continue
    return sessions


@app.post("/api/sessions")
def create_session():
    session_id = uuid.uuid4().hex[:12]
    save_session(session_id, [{"role": "system", "content": SYSTEM_PROMPT}])
    return {"session_id": session_id, "created_at": datetime.now().isoformat()}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    messages = load_session(session_id)
    chat_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]
    return {"session_id": session_id, "messages": chat_msgs}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    path = _session_path(session_id)
    if path.exists():
        path.unlink()
    cache.delete_session(session_id)
    if session_id in agents:
        del agents[session_id]
    return {"ok": True}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "chunks": rag.get_chunk_count(),
        "model": MODEL,
        "provider": LLM_PROVIDER,
        "engine": "langchain",
    }


# 静态文件
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print(f"\n  食鉴（LangChain 版）已启动 → http://127.0.0.1:{PORT}")
    print(f"  模型: {MODEL} | 知识库: {rag.get_chunk_count()} 块")
    print(f"  原版 → :8000 | LangChain 版 → :{PORT}\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
