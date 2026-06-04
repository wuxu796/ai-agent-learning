"""
食鉴 — 基于《中国营养科学全书》的 RAG 问答
"""
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from engine import BookRAG

load_dotenv()

# ==================== 1. 初始化 ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    print("❌ 请先设置环境变量 DEEPSEEK_API_KEY")
    exit(1)

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
rag = BookRAG()

# ==================== 2. 工具函数 ====================

ACTIVITY_LEVELS = {
    "久坐": 1.2, "轻度": 1.375, "中度": 1.55, "高度": 1.725, "运动员": 1.9,
}


def calculate_bmi(weight: float, height: float):
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


def calculate_tdee(gender: str, age: int, weight: float, height: float, activity: str):
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
    """搜索《中国营养科学全书》知识库"""
    if rag.get_chunk_count() == 0:
        return "知识库为空，请先运行 python build.py 构建知识库。"
    return rag.search_and_format(query, n_results=5)


# ==================== 3. 工具注册表 ====================
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
                "涵盖：基础营养、食物营养、公共营养、临床营养、妇幼营养、老年营养等七大卷。"
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

# ==================== 4. Agent 核心 ====================
SESSION_MESSAGES = []

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
    "- 回复中避免使用 ~ 符号，用'到'字代替。"
)


def run_one_turn(messages):
    response = client.chat.completions.create(
        model="deepseek-chat", messages=messages, tools=TOOLS, temperature=0.0
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"  [工具] {func_name}({func_args})")

            if func_name == "calculate_bmi":
                result = calculate_bmi(func_args["weight"], func_args["height"])
            elif func_name == "calculate_tdee":
                result = calculate_tdee(
                    func_args["gender"], func_args["age"],
                    func_args["weight"], func_args["height"], func_args["activity"],
                )
            elif func_name == "search_book":
                print("  [RAG] 检索中...")
                result = search_book(func_args["query"])
            else:
                result = f"未知工具: {func_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        final_response = client.chat.completions.create(
            model="deepseek-chat", messages=messages, temperature=0.7
        )
        assistant_reply = final_response.choices[0].message
        messages.append({"role": "assistant", "content": assistant_reply.content})
        return assistant_reply.content
    else:
        messages.append({"role": "assistant", "content": msg.content})
        return msg.content


# ==================== 5. Gradio 界面 ====================

def respond(message, history):
    global SESSION_MESSAGES
    SESSION_MESSAGES.append({"role": "user", "content": message})
    reply = run_one_turn(SESSION_MESSAGES)
    return reply


def reset_chat():
    global SESSION_MESSAGES
    SESSION_MESSAGES = [{"role": "system", "content": SYSTEM_PROMPT}]
    return []


def launch_web():
    import gradio as gr

    reset_chat()

    chunk_count = rag.get_chunk_count()
    knowledge_note = (
        "知识库已就绪" if chunk_count > 0
        else "⚠️ 知识库为空，请先运行 python build.py"
    )

    def chat_fn(message, history):
        reply = respond(message, None)
        return reply

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="食鉴",
        description=(
            f"算 BMI · 算每日消耗 · 搜营养知识\n"
            f"{knowledge_note}"
        ),
        textbox=gr.Textbox(placeholder="问点营养学问题，比如：维生素D的推荐摄入量是多少？"),
        examples=[
            "蛋白质的生理功能有哪些？",
            "维生素D缺乏会导致什么问题？",
            "膳食纤维对肠道健康的作用机制是什么？",
            "孕期需要补充哪些营养素？",
            "老年人蛋白质需要量和年轻人有什么不同？",
            "我身高175体重80，算一下BMI",
            "鸡胸肉和牛肉的营养成分有什么区别？",
        ],
    )

    demo.launch(server_name="127.0.0.1")


if __name__ == "__main__":
    launch_web()
