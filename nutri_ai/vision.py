"""
食鉴 2.0 — 视觉识别模块 (纯 Ollama 方案)
Qwen3-VL 单步识图 → JSON。速度优先，质量靠 prompt 校准。
"""
import base64
import io
import json
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from openai import (
    APIError,
    APIConnectionError,
    APITimeoutError,
    OpenAI,
)
from PIL import Image
from pydantic import BaseModel, field_validator

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b")

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama", timeout=60)

MAX_SIZE = 1280
JPEG_QUALITY = 75


# ── 异常定义 ───────────────────────────────


class VisionError(Exception):
    """视觉识别异常，retryable 标记调用方可选择重试。"""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


# ── 图片压缩 ───────────────────────────────


def _encode_b64(image_path: str) -> str:
    try:
        img = Image.open(image_path).convert("L")
    except Exception as e:
        raise VisionError(f"无法读取图片: {e}") from e

    w, h = img.size
    if max(w, h) > MAX_SIZE:
        ratio = MAX_SIZE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


PROMPT = """你是食品标签 OCR 专家。逐字识别图中所有文字，严格按以下 JSON 输出，不要任何额外文字：

{
  "product_name": "品名",
  "ingredients": ["配料", ...],
  "nutrition": {"per_100g": {"energy_kj":null,"fat_g":null,"sat_fat_g":null,"trans_fat_g":null,"carb_g":null,"sugar_g":null,"protein_g":null,"sodium_mg":null}, "serving_g":null},
  "additives": ["添加剂", ...]
}

必遵守：
1. 营养表优先取「每100克」列。若仅有「每份(X克)」，全部值÷X×100换算为每100克，serving_g记X
2. 能量：kJ直接填，kcal×4.184转kJ
3. carb_g取「碳水化合物」总值；sugar_g取「糖」或「蔗糖」子项值，标签未单标糖则填null
4. sat_fat_g取「饱和脂肪」值，trans_fat_g取「反式脂肪」值。两项独立，各填各的
5. 配料逐一拆开，去百分比和括号注释。维生素/矿物质/氨基酸属营养强化剂→ingredients；防腐剂/色素/香精/增稠剂等食品添加剂→additives
6. 看清的数字如实填（糖为0就填0），看不清填null，不编造
7. 纯JSON输出，禁止markdown包裹、禁止解释"""

OLLAMA_OPTS = {"keep_alive": "5m", "options": {"num_ctx": 2048}}


# ── 输出结构校验 ────────────────────────────


class NutritionPer100g(BaseModel):
    energy_kj: float | None = None
    fat_g: float | None = None
    sat_fat_g: float | None = None
    carb_g: float | None = None
    sugar_g: float | None = None
    protein_g: float | None = None
    sodium_mg: float | None = None
    trans_fat_g: float | None = None

    @field_validator("*", mode="before")
    @classmethod
    def coerce_numeric(cls, v):
        """容忍模型输出字符串数字（如 "2218" → 2218）。"""
        if v is None:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            raise ValueError(f"营养值必须为数字，收到: {v}")


class Nutrition(BaseModel):
    per_100g: NutritionPer100g = NutritionPer100g()
    serving_g: float | None = None


class VisionResult(BaseModel):
    product_name: str = ""
    ingredients: list[str] = []
    nutrition: Nutrition = Nutrition()
    additives: list[str] = []


def _validate_schema(data: dict) -> dict:
    """校验模型输出的 JSON 结构。缺失字段用默认值填充，类型错误抛 VisionError。"""
    try:
        return VisionResult(**data).model_dump()
    except Exception as e:
        raise VisionError(f"模型输出结构不合法: {e}", retryable=True) from e


# ── JSON 容错解析 ──────────────────────────


def _extract_json(text: str) -> dict:
    """从模型输出中提取 JSON。处理常见的格式问题：

    - "好的，以下是识别结果：{...}"   → 取第一个 { 到最后一个 }
    - "```json\n{...}\n```"           → 去掉 markdown 包裹
    - 模型附加的说明文字在 JSON 前后    → 自动裁剪
    """
    if not text or not text.strip():
        raise VisionError("模型返回了空内容，请重试", retryable=True)

    original = text.strip()

    # 策略1：标准清理后直接解析
    cleaned = original
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 策略2：模型输出混入了说明文字 — 从第一个 { 切到最后一个 }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        fragment = cleaned[start:end + 1]
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            # 片段内仍有语法问题，交给策略3修复
            pass
    else:
        fragment = None

    # 策略3：修复尾逗号（LLM 偶发），不做单引号替换——会误伤英文所有格
    target = fragment if fragment else cleaned
    try:
        fixed = re.sub(r",\s*}", "}", target)
        fixed = re.sub(r",\s*]", "]", fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    raise VisionError(
        f"模型返回了非 JSON 格式的内容（前100字）: {original[:100]}",
        retryable=True,
    )


# ── 主入口 ─────────────────────────────────


def scan_label(image_path: str, max_retries: int = 1) -> dict:
    """拍照识别配料表 → 结构化 JSON。

    错误处理覆盖三种场景：
    - Ollama 服务不可用 → VisionError(retryable=True)
    - 网络/超时          → VisionError(retryable=True)
    - 模型返回非 JSON     → VisionError(retryable=True)，最多重试 max_retries 次
    """
    image_b64 = _encode_b64(image_path)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            label = f"[Vision] {MODEL} 灰度 {MAX_SIZE}px"
            if attempt > 0:
                label += f" (重试 {attempt}/{max_retries})"
            print(f"{label} ...")

            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": PROMPT},
                    ],
                }],
                temperature=0.1,
                max_tokens=1000,
                extra_body=OLLAMA_OPTS,
            )

            text = resp.choices[0].message.content or ""
            elapsed = time.time() - t0
            print(f"[Vision] {elapsed:.0f}s")

            data = _extract_json(text)
            data = _validate_schema(data)
            _split_ingredients(data)
            _fix_per_serving(data)
            return data

        except VisionError as e:
            last_error = e
            print(f"[Vision] 格式异常，{'将重试' if attempt < max_retries else '已达最大重试次数'}")

        except APITimeoutError as e:
            last_error = VisionError(f"Ollama 响应超时: {e}", retryable=True)
            print(f"[Vision] 超时，{'将重试' if attempt < max_retries else '已达最大重试次数'}")

        except APIConnectionError as e:
            last_error = VisionError(
                f"无法连接 Ollama 服务 ({OLLAMA_URL}): {e}",
                retryable=True,
            )
            print(f"[Vision] 连接失败，{'将重试' if attempt < max_retries else '已达最大重试次数'}")

        except APIError as e:
            last_error = VisionError(f"Ollama API 错误: {e}", retryable=True)
            print(f"[Vision] API 错误: {e}")

    raise last_error or VisionError("未知错误")


def _split_ingredients(data: dict) -> None:
    """后处理：将模型可能粘连的配料字符串拆分为独立条目。

    模型即使按要求输出列表，也常把多个配料塞进一个字符串，例如：
      - ["小麦粉、白砂糖、植物油"] → ["小麦粉", "白砂糖", "植物油"]
      - ["食品添加剂（山梨酸钾、柠檬酸）"] → ["山梨酸钾", "柠檬酸"]
      - ["小麦粉(30%)"] → ["小麦粉"]

    处理逻辑：
      1. 中文/英文逗号、顿号、空格 → 拆分
      2. 去掉百分比 (30%)、（30%）等后缀
      3. 展开「食品添加剂（...）」括号内的内容
      4. 去重、去空、去纯数字残留
    """
    import re as _re

    SPLIT_RE = _re.compile(r"[,，、\s]+")
    PCT_RE = _re.compile(r"[（(]\s*\d+\s*[%％]\s*[)）]")
    ADDITIVE_WRAP_RE = _re.compile(r"食品添加剂[（(](.+)[)）]")

    def _split_and_clean(items: list) -> list[str]:
        result = []
        for item in items:
            if not isinstance(item, str):
                result.append(str(item))
                continue

            # 展开「食品添加剂（A、B、C）」
            m = ADDITIVE_WRAP_RE.search(item)
            if m:
                inner = m.group(1)
                result.extend(_split_and_clean([inner]))
                continue

            # 按分隔符拆开
            parts = [p.strip() for p in SPLIT_RE.split(item) if p.strip()]
            for p in parts:
                # 去百分比
                p = PCT_RE.sub("", p).strip()
                # 去括号注释（如"小麦粉(含麸质)" → "小麦粉"）
                p = _re.sub(r"[（(][^)）]*[)）]", "", p).strip()
                if p and not p.isdigit():
                    result.append(p)

        return result

    # 容错：ingredients 可能是 str（模型彻底没按 JSON 来）
    raw_ingredients = data.get("ingredients", [])
    if isinstance(raw_ingredients, str):
        raw_ingredients = [raw_ingredients]

    raw_additives = data.get("additives", [])
    if isinstance(raw_additives, str):
        raw_additives = [raw_additives]

    before_ing = len(raw_ingredients)
    before_add = len(raw_additives)

    data["ingredients"] = _split_and_clean(raw_ingredients)
    data["additives"] = _split_and_clean(raw_additives)

    if len(data["ingredients"]) != before_ing or len(data["additives"]) != before_add:
        print(
            f"[Fix] 配料拆分: {before_ing}+{before_add} → "
            f"{len(data['ingredients'])}+{len(data['additives'])}"
        )


def _fix_per_serving(data: dict) -> None:
    """若模型误取了每份值（有 serving_g 且值偏小），自动换算为每100g。"""
    nutrition = data.get("nutrition", {})
    serving_g = nutrition.get("serving_g")
    if not serving_g or serving_g == 100:
        return

    per_100g = nutrition.get("per_100g", {})
    if not per_100g:
        return

    # 检测：能量 < 1000kJ/100g 大概率是每份值（正常零食 1500-2500）
    energy = per_100g.get("energy_kj")
    if energy is None or energy >= 1000:
        return  # 已经是每100g值，不需要换算

    ratio = 100 / serving_g
    for key in ("energy_kj", "fat_g", "sat_fat_g", "trans_fat_g", "carb_g", "sugar_g", "protein_g", "sodium_mg"):
        val = per_100g.get(key)
        if val is not None:
            per_100g[key] = round(val * ratio, 1)
    print(f"[Fix] 检测到每份({serving_g}g)值，已换算为每100g (×{ratio:.1f})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python vision.py <图片路径> [输出json路径]")
        sys.exit(1)

    result = scan_label(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(sys.argv[1])[0] + "_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out}")
