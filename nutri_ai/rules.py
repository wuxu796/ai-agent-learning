"""食鉴规则引擎。

规则层只做确定性判断，不依赖 RAG 和 LLM。这样它可以被快速单测覆盖，
也方便后续升级成多 Agent 系统里的 Rule Agent。
"""

# NRV 基准：通用成年人每日推荐摄入量
NRV = {
    "energy_kj": 8400,
    "fat_g": 60,
    "sat_fat_g": 20,
    "trans_fat_g": 2,
    "carb_g": 300,
    "protein_g": 60,
    "sodium_mg": 2000,
}

NRV_LABEL = {
    "energy_kj": "能量",
    "fat_g": "脂肪",
    "sat_fat_g": "饱和脂肪",
    "trans_fat_g": "反式脂肪",
    "carb_g": "碳水",
    "protein_g": "蛋白质",
    "sodium_mg": "钠",
}

HIGH_RISK_KW = ["氢化", "反式脂肪", "亚硝酸", "苯甲酸", "苏丹红"]
MED_RISK_KW = ["阿斯巴甜", "亚硫酸", "焦糖色", "甜蜜素", "糖精", "味精"]
SAFE_KW = [
    "柠檬酸",
    "碳酸氢钠",
    "碳酸钙",
    "小苏打",
    "维生素",
    "抗坏血酸",
    "醋酸酯",
    "磷酸酯",
    "甜菊糖苷",
    "甘油脂肪酸酯",
    "单硬脂酸",
]

SUGAR_CONCERNS = {"糖尿病", "控糖"}
SODIUM_CONCERNS = {"高血压", "低钠"}
FAT_CONCERNS = {"高血脂", "减脂", "减脂控重", "控脂"}


def _nutrition(vision_json: dict) -> dict:
    return vision_json.get("nutrition", {}).get("per_100g", {})


def _num(nutrition: dict, key: str) -> float:
    try:
        return float(nutrition.get(key, 0) or 0)
    except (ValueError, TypeError):
        return 0.0


def _has_keyword(items: list, keywords: list[str]) -> bool:
    return any(any(k in str(item) for k in keywords) for item in items)


def _concerns(profile: dict | None) -> set[str]:
    if not profile:
        return set()
    return {str(item).strip() for item in profile.get("concerns", []) if str(item).strip()}


def _matched_profile_pressure(nutrition: dict, profile: dict | None) -> int:
    """只在用户关注点与食品短板匹配时加分，避免无差别抬高风险。"""
    concerns = _concerns(profile)
    if not concerns:
        return 0

    pressure = 0
    if concerns & SUGAR_CONCERNS and (
        _num(nutrition, "sugar_g") > 5 or _num(nutrition, "carb_g") > 50
    ):
        pressure += 1
    if concerns & SODIUM_CONCERNS and _num(nutrition, "sodium_mg") > 200:
        pressure += 1
    if concerns & FAT_CONCERNS and (
        _num(nutrition, "fat_g") > 10 or _num(nutrition, "sat_fat_g") > 5
    ):
        pressure += 1

    return min(pressure, 1)


def has_nutrition_data(vision_json: dict) -> bool:
    """检查 vision JSON 中是否至少有一项有效营养成分。"""
    nutrition = _nutrition(vision_json)
    if not nutrition:
        return False
    return any(v is not None for v in nutrition.values())


def classify_ingredients(vision_json: dict) -> list[dict]:
    """给每个配料评风险等级：safe / medium / high。"""
    items = []

    for ing in vision_json.get("ingredients", []):
        name = str(ing)
        if any(k in name for k in HIGH_RISK_KW):
            risk = "high"
        elif any(k in name for k in MED_RISK_KW):
            risk = "medium"
        elif any(k in name for k in SAFE_KW):
            risk = "safe"
        else:
            risk = "safe"
        items.append({"name": name, "risk": risk, "desc": ""})

    for add in vision_json.get("additives", []):
        name = str(add)
        if any(k in name for k in HIGH_RISK_KW):
            risk = "high"
        elif any(k in name for k in MED_RISK_KW):
            risk = "medium"
        else:
            risk = "safe"
        items.append({"name": name, "risk": risk, "desc": ""})

    return items


def compute_benchmarks(vision_json: dict) -> dict:
    """计算各营养指标占 NRV 的百分比。"""
    nutrition = _nutrition(vision_json)
    benchmarks = {}

    for key, nrv_value in NRV.items():
        raw = nutrition.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
            pct = round(val / nrv_value * 100)
            benchmarks[f"{key}_pct"] = f"{pct}%"
        except (ValueError, TypeError):
            continue

    return benchmarks


def calc_indicators(vision_json: dict) -> dict:
    """计算六维红绿灯。"""
    nutrition = _nutrition(vision_json)

    fat = _num(nutrition, "fat_g")
    sugar = _num(nutrition, "sugar_g")
    sodium = _num(nutrition, "sodium_mg")
    protein = _num(nutrition, "protein_g")
    trans = _num(nutrition, "trans_fat_g")

    if fat > 20:
        fat_light = "red"
    elif fat > 10:
        fat_light = "yellow"
    else:
        fat_light = "green"

    if sugar > 15:
        sugar_light = "red"
    elif sugar > 5:
        sugar_light = "yellow"
    else:
        sugar_light = "green"

    if sodium > 600:
        sodium_light = "red"
    elif sodium > 200:
        sodium_light = "yellow"
    else:
        sodium_light = "green"

    if protein > 12:
        protein_light = "green"
    elif protein > 5:
        protein_light = "yellow"
    else:
        protein_light = "red"

    additives = [a for a in vision_json.get("additives", []) if a]
    n_add = len(additives)
    if n_add > 5:
        add_light = "red"
    elif n_add >= 2:
        add_light = "yellow"
    else:
        add_light = "green"

    return {
        "fat": fat_light,
        "sugar": sugar_light,
        "sodium": sodium_light,
        "protein": protein_light,
        "additives": add_light,
        "trans_fat": "red" if trans > 0 else "green",
    }


def calc_risk(vision_json: dict, profile: dict | None) -> str:
    """综合风险评分：low / medium / high。"""
    nutrition = _nutrition(vision_json)
    ingredients = vision_json.get("ingredients", [])
    additives = [a for a in vision_json.get("additives", []) if a]

    score = 0
    if _has_keyword(ingredients + additives, HIGH_RISK_KW):
        score += 1
    if _num(nutrition, "fat_g") > 25:
        score += 1
    elif _num(nutrition, "fat_g") > 20:
        score += 0.5
    if _num(nutrition, "sodium_mg") > 600:
        score += 1
    if _num(nutrition, "sat_fat_g") > 12:
        score += 1
    if _num(nutrition, "sugar_g") > 15:
        score += 1
    if _num(nutrition, "trans_fat_g") > 0:
        score += 1
    if len(additives) > 5:
        score += 1
    elif len(additives) >= 2:
        score += 0.5
    score += _matched_profile_pressure(nutrition, profile)

    if score >= 3:
        return "high"
    if score >= 1:
        return "medium"
    return "low"


def _traffic_desc_parts(traffic_lights: dict) -> tuple[list[str], list[str]]:
    """把红绿灯拆成主要风险和提醒项，用于生成更准确的一句话。"""
    reds = {k for k, v in traffic_lights.items() if v == "red"}
    yellows = {k for k, v in traffic_lights.items() if v == "yellow"}

    risk_parts = []
    notes = []

    if "fat" in reds:
        risk_parts.append("高脂")
    elif "fat" in yellows:
        notes.append("脂肪偏高")

    if "sodium" in reds:
        risk_parts.append("高钠")
    elif "sodium" in yellows:
        notes.append("钠偏高")

    if "sugar" in reds:
        risk_parts.append("高糖")
    elif "sugar" in yellows:
        notes.append("糖偏高")

    if "trans_fat" in reds:
        risk_parts.append("含反式脂肪")

    if "additives" in reds:
        risk_parts.append("多添加剂")
    elif "additives" in yellows:
        notes.append("含添加剂")

    if "protein" in reds:
        notes.append("蛋白质较低")

    return risk_parts, notes


def generate_summary(
    risk_level: str,
    traffic_lights: dict,
    benchmarks: dict,
    nutrition_available: bool = True,
) -> str:
    """规则引擎生成一句话总结。"""
    if not nutrition_available:
        return "营养成分数据缺失，建议查看包装上的营养成分表后再判断"

    risk_parts, notes = _traffic_desc_parts(traffic_lights)
    desc_parts = risk_parts or notes
    desc = "、".join(desc_parts) if desc_parts else "营养表现平稳"

    if risk_level == "high":
        suffix = "建议严格控制摄入量，尽量少吃或不吃"
    elif risk_level == "medium":
        suffix = "偶尔解馋可以，不建议常吃"
    else:
        suffix = "日常食用无需担心"

    return f"{desc}食品，{suffix}"


def build_personalized_advice(
    personalized: dict,
    risk_level: str,
    benchmarks: dict,
    nutrition_available: bool = True,
) -> str:
    """基于 BMI/TDEE、风险等级和 NRV 基准生成个性化建议。"""
    parts = []

    bmi_info = personalized.get("bmi")
    if bmi_info:
        parts.append(f"你 BMI {bmi_info['value']}，属于{bmi_info['level']}范围")

    tdee_val = personalized.get("tdee")
    if tdee_val:
        parts.append(f"每日消耗约 {tdee_val} 千卡")

    if not nutrition_available:
        parts.append("营养成分数据缺失，建议查看包装上的营养成分表获取准确数据")
    else:
        high_items = []
        for key, pct_str in benchmarks.items():
            try:
                pct_val = int(pct_str.rstrip("%"))
            except (ValueError, AttributeError):
                continue
            if pct_val >= 30:
                base_key = key.replace("_pct", "")
                label = NRV_LABEL.get(base_key, base_key)
                high_items.append(f"{label}占NRV {pct_str}")

        if risk_level == "high":
            parts.append("这款食品风险较高，建议严格控制摄入")
        elif risk_level == "medium":
            if high_items:
                parts.append("这款食品" + "、".join(high_items[:2]) + "，建议适量食用")
            else:
                parts.append("建议适量食用，注意控制频率")
        else:
            parts.append("这款食品风险较低，日常食用无需担心")

    return "。".join(parts) + "。"
