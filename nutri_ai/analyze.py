"""食鉴 2.0 分析入口。

核心实现已拆到：
- rules.py：确定性规则引擎
- user_profile.py：用户画像与 BMI/TDEE
- llm_analysis.py：RAG 与 LLM 解读
- analysis_pipeline.py：整体编排

保留本文件是为了兼容旧脚本中的 `from analyze import full_analysis`。
"""

import json

from analysis_pipeline import full_analysis

__all__ = ["full_analysis"]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python analyze.py <vision_output.json> [profile.json]")
        print()
        print("示例:")
        print("  python analyze.py test_vision.json")
        print("  python analyze.py test_vision.json profile.json")
        print()
        print("环境变量:")
        print("  LLM_PROVIDER=ollama     → 使用本地 Ollama 模型")
        print("  LLM_PROVIDER=deepseek   → 使用 DeepSeek API（默认）")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    profile = None
    if len(sys.argv) > 2:
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            profile = json.load(f)

    result = full_analysis(data, profile)

    print(f"\n{'=' * 56}")
    print(f"  📦 产品: {result['product']['name']}")
    print(f"  📝 总结: {result['product']['summary']}")
    print(f"  ⚠️  风险: {result['verdict']['risk_level']}")
    tl = result["verdict"]["traffic_lights"]
    print(
        f"  🚦 红绿灯: 脂肪={tl['fat']} 糖={tl['sugar']} 钠={tl['sodium']} "
        f"蛋白质={tl['protein']} 添加剂={tl['additives']} 反式脂肪={tl['trans_fat']}"
    )
    print(f"{'=' * 56}")

    print("\n── 配料解读 ──")
    print(result["explanation"]["ingredient_analysis"])

    print("\n── 营养分析 ──")
    print(result["explanation"]["nutrition_analysis"])

    print(f"\n{'─' * 56}")
    print("[完整 JSON 输出]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
