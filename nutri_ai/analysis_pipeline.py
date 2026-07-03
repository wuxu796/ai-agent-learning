"""食鉴 2.0 分析编排层。"""

import time
from concurrent.futures import ThreadPoolExecutor

from llm_analysis import analyze_ingredients, analyze_nutrition, search_book_context
from user_profile import build_personalized_metrics
from rules import (
    build_personalized_advice,
    calc_indicators,
    calc_risk,
    classify_ingredients,
    compute_benchmarks,
    generate_summary,
    has_nutrition_data,
)


class AnalysisError(Exception):
    """分析流程异常，给 API 层转换成结构化错误。"""


def _validate_input(vision_json: dict) -> None:
    if not isinstance(vision_json, dict):
        raise AnalysisError("vision_json 必须是对象")
    if "nutrition" in vision_json and not isinstance(vision_json["nutrition"], dict):
        raise AnalysisError("nutrition 必须是对象")
    if "ingredients" in vision_json and not isinstance(vision_json["ingredients"], list):
        raise AnalysisError("ingredients 必须是数组")
    if "additives" in vision_json and not isinstance(vision_json["additives"], list):
        raise AnalysisError("additives 必须是数组")


def full_analysis(vision_json: dict, profile: dict | None = None) -> dict:
    """
    一键分析：规则引擎 -> RAG 搜书 -> LLM 并行解读 -> 组装输出。

    风险评级由规则引擎完成，LLM 只负责科普解释，避免把确定性判断交给模型。
    """
    _validate_input(vision_json)

    product_name = vision_json.get("product_name", "未知产品")
    nutrition_available = has_nutrition_data(vision_json)
    if not nutrition_available:
        print("[分析] ⚠️ 营养成分数据缺失，规则引擎和 LLM 将使用保守策略")

    ingredients = classify_ingredients(vision_json)
    benchmarks = compute_benchmarks(vision_json)
    traffic_lights = calc_indicators(vision_json)
    risk_level = calc_risk(vision_json, profile)
    summary = generate_summary(
        risk_level,
        traffic_lights,
        benchmarks,
        nutrition_available=nutrition_available,
    )
    rule_result = {
        "risk_level": risk_level,
        "traffic_lights": traffic_lights,
        "benchmarks": benchmarks,
        "summary": summary,
        "nutrition_available": nutrition_available,
    }

    print("[分析] RAG 搜书中...")
    t0 = time.time()
    book_context = search_book_context(vision_json, profile)
    print(f"[分析] RAG 搜索完成 ({time.time() - t0:.0f}s)")

    print("[分析] LLM 并行解读中（配料 ‖ 营养）...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_ing = executor.submit(analyze_ingredients, vision_json, book_context, rule_result)
        future_nut = executor.submit(analyze_nutrition, vision_json, profile, book_context, rule_result)
        ingredient_analysis = future_ing.result()
        nutrition_analysis = future_nut.result()
    print(f"[分析] LLM 解读完成 ({time.time() - t0:.0f}s)")

    personalized = build_personalized_metrics(profile)
    personalized["advice"] = build_personalized_advice(
        personalized,
        risk_level,
        benchmarks,
        nutrition_available=nutrition_available,
    )

    return {
        "product": {
            "name": product_name,
            "summary": summary,
        },
        "verdict": {
            "risk_level": risk_level,
            "traffic_lights": traffic_lights,
        },
        "ingredients": ingredients,
        "nutrition_detail": {
            "per_100g": vision_json.get("nutrition", {}).get("per_100g", {}),
            "benchmarks": benchmarks,
        },
        "personalized": personalized,
        "explanation": {
            "ingredient_analysis": ingredient_analysis,
            "nutrition_analysis": nutrition_analysis,
        },
    }
