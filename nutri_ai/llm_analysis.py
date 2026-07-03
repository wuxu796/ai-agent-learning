"""RAG 与 LLM 解读层。"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from engine import BookRAG
from user_profile import build_profile_text

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()

if LLM_PROVIDER == "ollama":
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    MODEL = OLLAMA_MODEL
    LLM_TIMEOUT = 90.0
    print(f"[LLM] 本地模型: {MODEL} @ {OLLAMA_URL}")
else:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("请设置 DEEPSEEK_API_KEY 或设置 LLM_PROVIDER=ollama")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    MODEL = "deepseek-chat"
    LLM_TIMEOUT = 30.0
    print("[LLM] DeepSeek API")

print("[RAG] 加载知识库...")
rag = BookRAG()
print(f"[RAG] 知识库就绪，共 {rag.get_chunk_count()} 块")


INGREDIENT_PROMPT = """你是一个懂食品的朋友，说话亲切但有依据。以下是某款食品的配料识别结果、规则引擎结论和营养学知识。

配料识别结果（JSON）：
{vision_json}

规则引擎结论：
{rule_context}

营养学知识：
{book_context}

请像聊天一样解读这份配料表，按以下节奏写：

1. **一句话定性**：先给整体判断（"这款整体还行""这个有点坑"这类大白话）
2. **配料怎么看**：前几位配料暗示了什么，越靠前含量越高，挑2-3个关键的讲
3. **添加剂一个个说**：每个添加剂是干嘛的、安不安全，有风险的实话实说，安全的明确说"放心"
4. **一句话收尾**：值不值得买、多久吃一次合适

写作规范：
- 风险等级、红绿灯结论已由规则引擎给出，你只能解释原因，不能推翻或重新评级
- 短句为主，像朋友发消息一样的自然节奏，读起来不费力
- 安全的添加剂直接消解焦虑："这个就是常见的XX，完全不用担心"
- 有风险的（如氢化油、高钠）不回避也不夸大，说清为什么要注意
- 涉及糖尿病、高血压、痛风、孕期等情况时，只给饮食注意建议，不做诊断或治疗建议
- 禁止提及任何信息来源（"根据全书""据文献"等），直接说结论
- 禁止使用 ~ 符号，用"到"字代替
- 输出控制在250字以内，宁短勿冗，但该讲清的添加剂不要跳过
"""

NUTRITION_PROMPT = """你是一个私人营养顾问，说话务实、给具体数字、不背教科书。以下是某款食品的营养成分、用户信息和规则引擎结论。

营养成分（每100g）：
{nutrition_text}

{profile_text}

规则引擎结论：
{rule_context}

营养学知识：
{book_context}

像给朋友做营养参谋一样，按以下节奏写：

1. **一句话定性**：这食品的营养水平怎么样（"高脂高钠的零食""蛋白质挺不错的加餐"这类）
2. **关键指标扫一遍**：脂肪、碳水、蛋白质、钠各处于什么水平（高/中/低），挑最突出的1-2项展开说，不搞平均主义
3. **你得吃多少**：结合用户体型，给一个具体的单次摄入建议数字（如"一次吃30g就够了""当早餐吃一整包问题不大"），并解释为什么
4. **特别提醒**：如果用户有健康关注（糖尿病、高血压等），针对性地提醒；没有就跳过这条

写作规范：
- 风险等级、红绿灯结论已由规则引擎给出，你只能解释原因，不能推翻或重新评级
- 如果缺少每份量、净含量或用户目标，不要编造"一包/一份"的建议，只能基于每100g做保守估算
- 涉及糖尿病、高血压、痛风、孕期等情况时，只给饮食注意建议，不做诊断或治疗建议
- 每个要点2-4句话，像在回复朋友微信，自然有温度
- 热量单位一律用千焦（kJ），不用千卡，不说"卡路里"
- 给具体数字而非模糊建议（"这包吃完占全天脂肪推荐量的40%"优于"脂肪含量较高"）
- 禁止使用 ~ 符号，用"到"字代替
- 输出控制在250字以内，宁短勿冗，但关键数据要说透
"""


def format_rule_context(rule_result: dict | None) -> str:
    """把规则引擎结果转成 LLM 可读的硬约束。"""
    if not rule_result:
        return "（未提供规则引擎结论，请只基于已识别数据做保守解释）"

    risk_level = rule_result.get("risk_level", "unknown")
    summary = rule_result.get("summary", "")
    traffic_lights = rule_result.get("traffic_lights", {})
    benchmarks = rule_result.get("benchmarks", {})
    nutrition_available = rule_result.get("nutrition_available", True)

    lines = [
        f"- 风险等级: {risk_level}",
        f"- 一句话总结: {summary or '无'}",
        f"- 红绿灯: {json.dumps(traffic_lights, ensure_ascii=False)}",
        f"- NRV占比: {json.dumps(benchmarks, ensure_ascii=False)}",
        f"- 营养成分是否完整: {'是' if nutrition_available else '否'}",
    ]
    return "\n".join(lines)


def search_book_context(vision_json: dict, profile: dict | None = None) -> str:
    """从配料、添加剂、营养指标和用户关注中构造检索上下文。"""
    if rag.get_chunk_count() == 0:
        return "（知识库为空）"

    keywords = []
    ingredients = vision_json.get("ingredients", [])
    additives = vision_json.get("additives", [])
    keywords.extend(additives[:5])
    keywords.extend(ingredients[:3])

    nutrition = vision_json.get("nutrition", {}).get("per_100g", {})

    def val(key):
        try:
            return float(nutrition.get(key, 0) or 0)
        except (ValueError, TypeError):
            return 0.0

    if val("fat_g") > 20:
        keywords.append("高脂肪食品")
    if val("sodium_mg") > 500:
        keywords.append("高钠饮食")
    if val("sat_fat_g") > 10:
        keywords.append("饱和脂肪酸与心血管")
    if val("trans_fat_g") > 0:
        keywords.append("反式脂肪酸健康危害")

    if profile:
        for concern in profile.get("concerns", []):
            keywords.append(concern + " 饮食注意事项")

    query = "食品添加剂 " + " ".join(keywords) if keywords else "零食营养成分每日推荐摄入量"
    return rag.search_and_format(query, n_results=5)


def call_llm(system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
    """LLM 调用封装，失败时返回兜底文本而不是打断主流程。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                timeout=LLM_TIMEOUT,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == 0:
                print(f"[LLM] 调用失败，1 秒后重试: {e}")
                time.sleep(1)
            else:
                print(f"[LLM] 重试仍失败，使用兜底文本: {e}")

    return (
        "AI 服务暂时不可用，请稍后重试。\n\n"
        "规则引擎已完成自动分析，具体结果请查看上方的风险等级和红绿灯标识。"
    )


def analyze_ingredients(
    vision_json: dict,
    book_context: str,
    rule_result: dict | None = None,
) -> str:
    """配料表科普解读。"""
    system_prompt = INGREDIENT_PROMPT.format(
        vision_json=json.dumps(vision_json, ensure_ascii=False, indent=2),
        rule_context=format_rule_context(rule_result),
        book_context=book_context,
    )
    return call_llm(
        system_prompt,
        "请分析这份配料表，让消费者放心地了解自己吃的东西。",
        temperature=0.3,
    )


def analyze_nutrition(
    vision_json: dict,
    profile: dict | None,
    book_context: str,
    rule_result: dict | None = None,
) -> str:
    """营养成分个性化分析。"""
    nutrition = vision_json.get("nutrition", {}).get("per_100g", {})
    nutrition_text = "\n".join(
        f"- {k}: {v}" for k, v in nutrition.items() if v is not None
    )
    if not nutrition_text.strip():
        return "（营养成分表未识别到数据，无法分析）"

    system_prompt = NUTRITION_PROMPT.format(
        nutrition_text=nutrition_text,
        profile_text=build_profile_text(profile),
        rule_context=format_rule_context(rule_result),
        book_context=book_context,
    )
    return call_llm(
        system_prompt,
        "请分析这款食品的营养成分，给出具体的食用建议。",
        temperature=0.3,
    )
