"""
批量测试脚本：picture 文件夹 6 张配料表 → vision → analyze
输出汇总到 batch_results.json

用法：cd nutri_ai && python tests/batch_test.py
"""
import json
import os
import sys
import time
from pathlib import Path

# 把项目根目录加入 sys.path，确保能导入 vision / analyze 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── 检查依赖 ──
try:
    from vision import scan_label
except ImportError as e:
    print(f"❌ 无法导入 vision 模块: {e}")
    sys.exit(1)

try:
    from analyze import full_analysis
except ImportError as e:
    print(f"❌ 无法导入 analyze 模块: {e}")
    sys.exit(1)

# ── 加载用户画像 ──
profile_path = TESTS_DIR / "test_profile.json"
profile = None
if os.path.exists(profile_path):
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
    print(f"📋 已加载用户画像: {profile_path}")
else:
    print("⚠️  无用户画像，将使用通用标准")

# ── 收集图片 ──
picture_dir = Path("pictures")
images = sorted(picture_dir.glob("*"))
images = [p for p in images if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]

if not images:
    print("❌ pictures/ 目录下没有图片")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"  批量测试: {len(images)} 张图片")
print(f"  流程: vision (OCR+结构化) → analyze (规则引擎+RAG+LLM)")
print(f"{'='*60}\n")

results = []
total_start = time.time()

for i, img_path in enumerate(images, 1):
    print(f"\n{'─'*60}")
    print(f"  [{i}/{len(images)}] {img_path.name}")
    print(f"{'─'*60}")

    # ── Step 1: Vision ──
    step_start = time.time()
    print(f"  👁️  Vision 识别中...")
    try:
        vision_json = scan_label(str(img_path))
        vision_time = time.time() - step_start
        print(f"  ✅ Vision 完成 ({vision_time:.0f}s)")
        print(f"     产品: {vision_json.get('product_name', '未知')}")
        print(f"     配料: {len(vision_json.get('ingredients', []))} 项")
        print(f"     添加剂: {len(vision_json.get('additives', []))} 项")
        nut = vision_json.get("nutrition", {}).get("per_100g", {})
        if nut:
            print(f"     营养成分: 能量={nut.get('energy_kj')} 脂肪={nut.get('fat_g')} "
                  f"钠={nut.get('sodium_mg')}")
    except Exception as e:
        print(f"  ❌ Vision 失败: {e}")
        results.append({
            "image": img_path.name,
            "vision_error": str(e),
            "vision_time_s": time.time() - step_start,
        })
        continue

    # 保存 vision 结果到 tests/ 目录
    vision_out = str(TESTS_DIR / (img_path.stem + "_vision.json"))
    with open(vision_out, "w", encoding="utf-8") as f:
        json.dump(vision_json, f, ensure_ascii=False, indent=2)

    # ── Step 2: Analyze ──
    step_start = time.time()
    print(f"  🔬 Analyze 分析中...")
    try:
        analysis = full_analysis(vision_json, profile)
        analyze_time = time.time() - step_start
        print(f"  ✅ Analyze 完成 ({analyze_time:.0f}s)")
        print(f"     风险: {analysis['verdict']['risk_level']}")
        print(f"     红绿灯: {analysis['verdict']['traffic_lights']}")
        print(f"     总结: {analysis['product']['summary']}")
    except Exception as e:
        print(f"  ❌ Analyze 失败: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "image": img_path.name,
            "vision_ok": True,
            "analyze_error": str(e),
            "analyze_time_s": time.time() - step_start,
        })
        continue

    # 保存 analyze 结果到 tests/ 目录
    analyze_out = str(TESTS_DIR / (img_path.stem + "_analysis.json"))
    with open(analyze_out, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    # ── 收集结果 ──
    results.append({
        "image": img_path.name,
        "vision_time_s": round(vision_time, 1),
        "analyze_time_s": round(analyze_time, 1),
        "product_name": analysis["product"]["name"],
        "risk_level": analysis["verdict"]["risk_level"],
        "traffic_lights": analysis["verdict"]["traffic_lights"],
        "summary": analysis["product"]["summary"],
        "ingredient_count": len(analysis["ingredients"]),
        "benchmarks": analysis["nutrition_detail"]["benchmarks"],
        "vision_file": vision_out,
        "analysis_file": analyze_out,
    })

# ── 汇总 ──
total_time = time.time() - total_start
print(f"\n\n{'='*60}")
print(f"  批量测试完成 ({total_time:.0f}s)")
print(f"{'='*60}")

print(f"\n{'图片':<15s} {'产品':<20s} {'风险':<8s} {'红绿灯'}")
print(f"{'-'*15} {'-'*20} {'-'*8} {'-'*40}")
for r in results:
    if "vision_error" in r:
        print(f"{r['image']:<15s} ❌ Vision错误")
    elif "analyze_error" in r:
        print(f"{r['image']:<15s} ❌ Analyze错误")
    else:
        tl = r["traffic_lights"]
        tl_str = f"脂={tl['fat']} 糖={tl['sugar']} 钠={tl['sodium']} 蛋={tl['protein']} 添={tl['additives']}"
        print(f"{r['image']:<15s} {r['product_name']:<20s} {r['risk_level']:<8s} {tl_str}")

# 保存汇总到 tests/ 目录
with open(TESTS_DIR / "batch_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n📄 详细结果: batch_results.json")
print(f"📄 各图片 vision JSON: p*_vision.json")
print(f"📄 各图片 analysis JSON: p*_analysis.json")
