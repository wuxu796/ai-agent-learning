"""分析编排层的稳定测试。

用假 RAG/LLM 替身测试结构和规则传递，不调用真实模型。
"""

import importlib
import sys
import types
import unittest


def vision_fixture():
    return {
        "product_name": "测试食品",
        "ingredients": ["小麦粉", "植物油"],
        "additives": ["柠檬酸"],
        "nutrition": {
            "per_100g": {
                "energy_kj": 1800,
                "fat_g": 22,
                "sat_fat_g": 5,
                "trans_fat_g": 0,
                "carb_g": 55,
                "sugar_g": 4,
                "protein_g": 6,
                "sodium_mg": 650,
            },
            "serving_g": 30,
        },
    }


class AnalysisPipelineTest(unittest.TestCase):
    def setUp(self):
        self.original_llm_analysis = sys.modules.get("llm_analysis")
        fake_llm = types.ModuleType("llm_analysis")
        fake_llm.search_book_context = lambda vision_json, profile=None: "fake context"
        fake_llm.analyze_ingredients = (
            lambda vision_json, book_context, rule_result=None:
            f"配料解释:{rule_result['risk_level']}:{rule_result['traffic_lights']['fat']}"
        )
        fake_llm.analyze_nutrition = (
            lambda vision_json, profile, book_context, rule_result=None:
            f"营养解释:{rule_result['summary']}"
        )
        sys.modules["llm_analysis"] = fake_llm
        sys.modules.pop("analysis_pipeline", None)
        self.pipeline = importlib.import_module("analysis_pipeline")

    def tearDown(self):
        sys.modules.pop("analysis_pipeline", None)
        if self.original_llm_analysis is None:
            sys.modules.pop("llm_analysis", None)
        else:
            sys.modules["llm_analysis"] = self.original_llm_analysis

    def test_full_analysis_uses_rules_and_preserves_contract(self):
        result = self.pipeline.full_analysis(
            vision_fixture(),
            profile={"height": 170, "weight": 65, "age": 30, "gender": "男", "activity": "中度"},
        )

        self.assertEqual(result["product"]["name"], "测试食品")
        self.assertEqual(result["verdict"]["risk_level"], "medium")
        self.assertEqual(result["verdict"]["traffic_lights"]["fat"], "red")
        self.assertEqual(result["verdict"]["traffic_lights"]["sodium"], "red")
        self.assertIn("高脂", result["product"]["summary"])
        self.assertIn("配料解释:medium:red", result["explanation"]["ingredient_analysis"])
        self.assertIn("营养解释:", result["explanation"]["nutrition_analysis"])
        self.assertEqual(result["personalized"]["bmi"]["value"], 22.5)

    def test_invalid_input_rejected_before_llm(self):
        with self.assertRaises(self.pipeline.AnalysisError):
            self.pipeline.full_analysis({"ingredients": "小麦粉"})


if __name__ == "__main__":
    unittest.main()
