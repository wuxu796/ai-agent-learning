"""
食鉴 — FastAPI 后端
提供：多会话聊天、历史持久化、工具编排
支持：DeepSeek API / Ollama 本地模型，通过环境变量 LLM_PROVIDER 切换
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
from openai import OpenAI

from engine import BookRAG

load_dotenv()

# ==================== LLM 提供者 ====================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()

if LLM_PROVIDER == "ollama":
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    MODEL = OLLAMA_MODEL
    print(f"[LLM] 本地模型: {MODEL} @ {OLLAMA_URL}")
else:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("请设置 DEEPSEEK_API_KEY 或设置 LLM_PROVIDER=ollama")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    MODEL = "deepseek-chat"
    print(f"[LLM] DeepSeek API")

# ==================== 初始化 ====================

rag = BookRAG()

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="食鉴")

# ==================== 工具函数 ====================

ACTIVITY_LEVELS = {
    "久坐": 1.2, "轻度": 1.375, "中度": 1.55, "高度": 1.725, "运动员": 1.9,
}


def calculate_bmi(weight: float, height: float) -> str:
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


def calculate_tdee(gender: str, age: int, weight: float, height: float, activity: str) -> str:
    if age <= 0 or weight <= 0 or height <= 0:
        return "年龄、体重、身高必须大于 0。"
    if gender == "男":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    elif gender == "女":
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        return "性别请输入'男'或'女'。"
    for level, ratio in ACTIVITY_LEVELS.items():
        if level in activity or activity in level:
            break
    else:
        return f"活动量'{activity}'不在可选范围内，试试：{', '.join(ACTIVITY_LEVELS.keys())}"
    tdee = round(bmr * ratio)
    return (
        f"基础代谢（BMR）：{round(bmr)} 千卡/天\n"
        f"每日消耗（TDEE）：{tdee} 千卡/天（{activity}模式，系数{ratio}）\n"
        f"减脂建议：每天摄入约 {tdee - 500} 千卡\n"
        f"增肌建议：每天摄入约 {tdee + 300} 千卡"
    )


def search_book(query: str) -> str:
    if rag.get_chunk_count() == 0:
        return "知识库为空，请先运行 python build.py 构建知识库。"
    return rag.search_and_format(query, n_results=5)


# ==================== 工具注册表 ====================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": "计算身体质量指数（BMI）。用户想知道体重是否正常时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight": {"type": "number", "description": "体重（公斤）"},
                    "height": {"type": "number", "description": "身高（厘米）"},
                },
                "required": ["weight", "height"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_tdee",
            "description": "计算每日总能量消耗（TDEE）。用户问'每天消耗多少'、'减脂吃多少'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "gender": {"type": "string", "description": "性别：'男'或'女'"},
                    "age": {"type": "integer", "description": "年龄"},
                    "weight": {"type": "number", "description": "体重（公斤）"},
                    "height": {"type": "number", "description": "身高（厘米）"},
                    "activity": {"type": "string", "description": "活动量：久坐/轻度/中度/高度/运动员"},
                },
                "required": ["gender", "age", "weight", "height", "activity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_book",
            "description": (
                "搜索《中国营养科学全书》知识库。"
                "当用户问到营养学理论、营养素功能、代谢机制、膳食指南、"
                "缺乏症、食物成分、各类人群营养需求等专业知识时，务必调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "你叫食鉴，是一个营养健康助手。\n"
    "你可以帮用户：\n"
    "1. 计算 BMI（身体质量指数）\n"
    "2. 计算 TDEE（每日消耗热量）\n"
    "3. 回答各类营养学问题\n\n"
    "回答规则：\n"
    "- 风格简洁、专业，用生活化的语言解释专业概念。\n"
    "- 回答营养学问题时，先调用 search_book 检索相关知识再回答。\n"
    "- 如果用户只是想算BMI或TDEE，直接用计算工具即可。\n"
    "- 禁止在回复中提及任何信息来源，包括但不限于：\"根据全书\"、\"书中记载\"、"
    "\"参考资料\"、\"文献显示\"、\"研究表明\"等。直接回答问题本身。\n"
    "- 回复中避免使用 ~ 符号，用'到'字代替。\n"
    "- 调用工具是系统行为，禁止在回复正文中输出 <tool_calls>、<invoke>、<function> "
    "等标签或工具调用语法。直接给出自然语言回答。"
)

# ==================== 会话管理 ====================


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def load_session(session_id: str) -> list[dict]:
    path = _session_path(session_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def save_session(session_id: str, messages: list[dict]):
    _session_path(session_id).write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ==================== Agent ====================


def _clean_response(text: str) -> str:
    """移除模型误输出的工具调用标签"""
    text = re.sub(r"<tool_calls>[^<]*</tool_calls>", "", text, flags=re.DOTALL)
    text = re.sub(r"<invoke[^>]*>[\s\S]*?</invoke>", "", text)
    text = re.sub(r"</?(?:tool_calls|invoke|function|parameter)[^>]*>", "", text)
    return text.strip()


def run_one_turn(session_id: str, user_message: str) -> str:
    messages = load_session(session_id)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS, temperature=0.0
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            if func_name == "calculate_bmi":
                result = calculate_bmi(func_args["weight"], func_args["height"])
            elif func_name == "calculate_tdee":
                result = calculate_tdee(
                    func_args["gender"], func_args["age"],
                    func_args["weight"], func_args["height"], func_args["activity"],
                )
            elif func_name == "search_book":
                result = search_book(func_args["query"])
            else:
                result = f"未知工具: {func_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        final_response = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.7
        )
        assistant_reply = final_response.choices[0].message
        messages.append({"role": "assistant", "content": assistant_reply.content})
        reply = assistant_reply.content
    else:
        messages.append({"role": "assistant", "content": msg.content})
        reply = msg.content

        # 保底策略：本地模型 Function Calling 不稳定，自动补算 + 搜书
        if LLM_PROVIDER == "ollama":
            fallback_parts = []

            # 1. 从用户消息提取数据，自动计算 BMI / TDEE
            h_match = re.search(r"身高\s*(\d{2,3})", user_message)
            w_match = re.search(r"体重\s*(\d{2,3})", user_message)

            if h_match and w_match:
                height = int(h_match.group(1))
                weight = int(w_match.group(1))
                fallback_parts.append(f"[BMI自动计算]\n{calculate_bmi(weight, height)}")

                # TDEE 还需要性别、年龄、活动量
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
                            f"[TDEE自动计算]\n"
                            + calculate_tdee(gender, age, weight, height, activity)
                        )

            # 2. 自动搜索知识库（去掉数字，用纯净关键词）
            clean_query = re.sub(r"身高\s*\d{2,3}", "", user_message)
            clean_query = re.sub(r"体重\s*\d{2,3}", "", clean_query)
            clean_query = re.sub(r"\d{1,3}\s*岁", "", clean_query)
            clean_query = re.sub(r"[男女]生?", "", clean_query)
            book_result = search_book(clean_query.strip() or user_message)
            if book_result and "没有找到相关信息" not in book_result:
                fallback_parts.append(f"[知识库检索]\n{book_result}")

            if fallback_parts:
                # 注意：不包含模型之前编造的回答，避免它顺着幻觉继续说
                temp_messages = messages[:-1].copy()
                temp_messages.append({
                    "role": "user",
                    "content": (
                        "以下数据由系统自动计算和检索，**请你必须使用这些数据来回答**，"
                        "不要自己编造任何数字。如果某项数据没有提供，不要替用户猜测。\n\n"
                        + "\n\n".join(fallback_parts)
                    ),
                })
                fallback = client.chat.completions.create(
                    model=MODEL, messages=temp_messages, temperature=0.7,
                )
                reply = fallback.choices[0].message.content
                messages[-1] = {"role": "assistant", "content": reply}

    save_session(session_id, messages)
    return _clean_response(reply)


# ==================== 流式生成器（实验性，已注释） ====================

# import time  # 流式需要
# from fastapi.responses import StreamingResponse  # 流式需要
#
# async def run_one_turn_stream(session_id: str, user_message: str):
#     """SSE 流式版本，工具决策非流式，最终合成逐字返回。DeepSeek chunk 偏大，效果不明显。"""
#     messages = load_session(session_id)
#     messages.append({"role": "user", "content": user_message})
#     response = client.chat.completions.create(
#         model=MODEL, messages=messages, tools=TOOLS, temperature=0.0
#     )
#     msg = response.choices[0].message
#     if msg.tool_calls:
#         # ... 工具执行（同 run_one_turn）...
#         stream = client.chat.completions.create(
#             model=MODEL, messages=messages, temperature=0.7, stream=True
#         )
#         full_reply = ""
#         for chunk in stream:
#             delta = chunk.choices[0].delta
#             if delta.content:
#                 full_reply += delta.content
#                 yield f"data: {json.dumps({'token': delta.content}, ensure_ascii=False)}\n\n"
#                 # time.sleep(0.01)
#         messages.append({"role": "assistant", "content": _clean_response(full_reply)})
#         save_session(session_id, messages)
#     yield "data: [DONE]\n\n"
#
#
# @app.post("/api/chat/stream")
# async def chat_stream(req: ChatRequest):
#     if not req.message.strip():
#         raise HTTPException(400, "消息不能为空")
#     return StreamingResponse(
#         run_one_turn_stream(req.session_id, req.message),
#         media_type="text/event-stream",
#     )


# ==================== API ====================


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "消息不能为空")
    reply = run_one_turn(req.session_id, req.message)
    return ChatResponse(session_id=req.session_id, reply=reply)


@app.get("/api/sessions")
def list_sessions():
    """列出所有历史会话"""
    sessions = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            messages = json.loads(path.read_text(encoding="utf-8"))
            # 找第一条用户消息作为预览
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
    return {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
    }


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    messages = load_session(session_id)
    chat = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]
    return {"session_id": session_id, "messages": chat}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    path = _session_path(session_id)
    if path.exists():
        path.unlink()
    return {"ok": True}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "chunks": rag.get_chunk_count(),
        "model": MODEL,
        "provider": LLM_PROVIDER,
    }


# 静态文件（前端页面）
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print(f"\n  食鉴已启动 → http://127.0.0.1:8000")
    print(f"  模型: {MODEL} | 知识库: {rag.get_chunk_count()} 块\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
