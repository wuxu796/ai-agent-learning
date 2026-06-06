"""
============ LangChain 入门 ============

对比实验：用 LangChain 重写食鉴的工具编排
手写版（server.py）：~80 行 if-elif 判断
LangChain 版：用 create_tool_calling_agent 自动编排

运行前的准备工作：
  pip install langchain langchain-openai python-dotenv
  （在 .env 里配好 DEEPSEEK_API_KEY）
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ====================================================
# 第一步：定义 LLM（和手写版一样，只是换了个包装）
# ====================================================

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

print("[1] LLM 包装完成 — DeepSeek API\n")

# ====================================================
# 第二步：定义工具（比手写版短一半！）
# ====================================================
# 对比：手写版要写 function schemas 字典 + JSON schema + 手动 type 转换
#       LangChain：加个 @tool 装饰器完事，函数签名自动生成参数 schema

from langchain.tools import tool

ACTIVITY_MAP = {"久坐": 1.2, "轻度": 1.375, "中度": 1.55, "高度": 1.725, "运动员": 1.9}


@tool
def calculate_bmi(weight: float, height: float) -> str:
    """计算身体质量指数 BMI。用户提供体重(公斤)和身高(厘米)时调用。"""
    height_m = height / 100
    bmi = round(weight / (height_m**2), 1)
    if bmi < 18.5:
        return f"BMI = {bmi}（偏瘦），建议增重。"
    elif bmi < 24:
        return f"BMI = {bmi}（正常），继续保持！"
    elif bmi < 28:
        return f"BMI = {bmi}（偏胖），建议控制饮食。"
    else:
        return f"BMI = {bmi}（肥胖），建议系统减脂。"


@tool
def calculate_tdee(gender: str, age: int, weight: float, height: float, activity: str) -> str:
    """计算每日总能量消耗 TDEE。用户想知道每天消耗多少热量时调用。
    gender: '男' 或 '女'  activity: 久坐/轻度/中度/高度/运动员"""
    if gender == "男":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    ratio = ACTIVITY_MAP.get(activity, 1.2)
    tdee = round(bmr * ratio)
    return f"TDEE = {tdee} 千卡/天（{activity}），减脂建议每天摄入 {tdee - 500} 千卡。"


@tool
def search_book(query: str) -> str:
    """搜索营养学知识库。用户问到营养学理论、食品成分、膳食指南时务必调用。
    注意：如果用户只是想算 BMI/TDEE，不需要调这个工具。"""
    # 这里模拟搜索结果（真实项目里接 ChromaDB）
    return f'[知识库] 关于"{query}"的检索结果：中国居民膳食指南建议...（模拟数据）'


tools = [calculate_bmi, calculate_tdee, search_book]
print("[2] 工具定义完成 — 3 个 @tool，无需手写 JSON schema\n")

# ====================================================
# 第三步：Agent 自动编排（替代手写 80 行 if-elif）
# ====================================================

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

# 一句话创建 Agent：自动判断调哪个工具、提取参数、拼接结果
prompt = ChatPromptTemplate.from_messages([
    ("system", "你叫食鉴，是一个营养健康助手。回答简洁专业。"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("[3] Agent 创建完成 — 一行替代手写 if-elif-elif-elif\n")

# ====================================================
# 第四步：演示（注意自动判断工具的过程）
# ====================================================

print("=" * 50)
print("测试 1：纯计算（只调一个工具）")
print("=" * 50)
result = executor.invoke({"input": "我身高175，体重72，帮我算BMI"})
print(f"结果: {result['output']}\n")

print("=" * 50)
print("测试 2：需要搜书（工具选择正确吗？）")
print("=" * 50)
result = executor.invoke({"input": "维生素C有什么作用"})
print(f"结果: {result['output']}\n")

print("=" * 50)
print("测试 3：闲聊（不需要调任何工具）")
print("=" * 50)
result = executor.invoke({"input": "你好！你是谁？"})
print(f"结果: {result['output']}\n")

print("=" * 50)
print("对比结论")
print("=" * 50)
print("""
手写版你需要做的：
  ├─ 写 JSON function schema（~40行）
  ├─ 解析 tool_calls.arguments（json.loads）
  ├─ if func == 'xxx' → elif 'yyy' → elif 'zzz'（~20行）
  ├─ 拼 messages（tool role + tool_call_id）
  ├─ 发起第二次 LLM 调用
  └─ 结果拼回 messages

LangChain 版：
  agent = create_tool_calling_agent(llm, tools, prompt)
  executor.invoke({"input": "..."})
  → 一行，自动完成上面所有步骤 ✅
""")
