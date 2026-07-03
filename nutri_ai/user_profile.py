"""用户画像与健康计算工具。"""

ACTIVITY_LEVELS = {
    "久坐": 1.2,
    "轻度": 1.375,
    "中度": 1.55,
    "高度": 1.725,
    "运动员": 1.9,
}


def calc_bmi(weight_kg: float, height_cm: float) -> dict:
    """计算 BMI 并返回等级。"""
    h = height_cm / 100
    bmi = round(weight_kg / (h ** 2), 1)
    if bmi < 18.5:
        level = "偏瘦"
    elif bmi < 24:
        level = "正常"
    elif bmi < 28:
        level = "偏胖"
    else:
        level = "肥胖"
    return {"value": bmi, "level": level}


def calc_tdee(gender: str, weight: float, height: float, age: int, activity: str) -> dict:
    """计算 TDEE 和减脂/增肌建议。"""
    if gender == "男":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    ratio = ACTIVITY_LEVELS.get(activity, 1.2)
    tdee = round(bmr * ratio)
    return {"bmr": round(bmr), "tdee": tdee, "cut": tdee - 500, "bulk": tdee + 300}


def build_profile_text(profile: dict | None) -> str:
    """把用户画像转成 LLM prompt 可读文本。"""
    if not profile:
        return "（用户未填写个人数据，按通用成年人标准给出建议）"

    parts = []
    h = profile.get("height")
    w = profile.get("weight")
    if h and w:
        bmi = calc_bmi(w, h)
        parts.append(f"身高 {h}cm，体重 {w}kg，BMI = {bmi['value']}（{bmi['level']}）")

    age = profile.get("age")
    gender = profile.get("gender")
    activity = profile.get("activity")
    if all([gender, age, h, w, activity]):
        tdee = calc_tdee(gender, w, h, age, activity)
        parts.append(f"每日消耗约 {tdee['tdee']} 千卡，减脂建议摄入 {tdee['cut']} 千卡")

    concerns = profile.get("concerns", [])
    if concerns:
        parts.append(f"健康关注：{'、'.join(concerns)}")

    return "用户信息：" + "；".join(parts) + "。请根据这些信息给出个性化建议。"


def build_personalized_metrics(profile: dict | None) -> dict:
    """根据用户画像计算 BMI/TDEE。"""
    personalized = {}
    if not profile:
        return personalized

    h = profile.get("height")
    w = profile.get("weight")
    if h and w:
        personalized["bmi"] = calc_bmi(w, h)

    gender = profile.get("gender")
    age = profile.get("age")
    activity = profile.get("activity")
    if all([gender, age, h, w, activity]):
        tdee = calc_tdee(gender, w, h, age, activity)
        personalized["tdee"] = tdee["tdee"]

    return personalized
