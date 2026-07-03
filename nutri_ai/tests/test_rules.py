"""规则引擎轻量测试。

运行方式：
  cd nutri_ai && python -m unittest tests.test_rules

这些测试不调用视觉模型、RAG、LLM，用来快速保护确定性规则。
"""

import unittest

from user_profile import calc_bmi, calc_tdee
from rules import (
    calc_indicators,
    calc_risk,
    classify_ingredients,
    compute_benchmarks,
    generate_summary,
    has_nutrition_data,
)


def vision_fixture(**nutrition):
    return {
        "product_name": "测试食品",
        "ingredients": ["小麦粉", "氢化植物油"],
        "additives": ["阿斯巴甜", "柠檬酸"],
        "nutrition": {
            "per_100g": {
                "energy_kj": nutrition.get("energy_kj", 2100),
                "fat_g": nutrition.get("fat_g", 30),
                "sat_fat_g": nutrition.get("sat_fat_g", 13),
                "trans_fat_g": nutrition.get("trans_fat_g", 0),
                "carb_g": nutrition.get("carb_g", 50),
                "sugar_g": nutrition.get("sugar_g", 8),
                "protein_g": nutrition.get("protein_g", 6),
                "sodium_mg": nutrition.get("sodium_mg", 700),
            }
        },
    }


def neutral_fixture(**nutrition):
    return {
        "product_name": "低风险测试食品",
        "ingredients": ["小麦粉", "水"],
        "additives": [],
        "nutrition": {
            "per_100g": {
                "energy_kj": nutrition.get("energy_kj", 800),
                "fat_g": nutrition.get("fat_g", 6),
                "sat_fat_g": nutrition.get("sat_fat_g", 0),
                "trans_fat_g": nutrition.get("trans_fat_g", 0),
                "carb_g": nutrition.get("carb_g", 20),
                "sugar_g": nutrition.get("sugar_g", 0),
                "protein_g": nutrition.get("protein_g", 8),
                "sodium_mg": nutrition.get("sodium_mg", 50),
            }
        },
    }


class RuleEngineTest(unittest.TestCase):
    def test_high_fat_sodium_and_risk(self):
        data = vision_fixture()

        lights = calc_indicators(data)
        self.assertEqual(lights["fat"], "red")
        self.assertEqual(lights["sodium"], "red")
        self.assertEqual(lights["sugar"], "yellow")
        self.assertEqual(lights["protein"], "yellow")

        risk = calc_risk(data, profile=None)
        self.assertEqual(risk, "high")

        summary = generate_summary(risk, lights, compute_benchmarks(data))
        self.assertIn("高脂", summary)
        self.assertIn("高钠", summary)

    def test_trans_fat_is_red(self):
        data = vision_fixture(fat_g=5, sodium_mg=100, sat_fat_g=0, trans_fat_g=0.1)

        lights = calc_indicators(data)
        self.assertEqual(lights["trans_fat"], "red")
        self.assertEqual(calc_risk(data, profile=None), "medium")

    def test_ingredient_keyword_classification(self):
        items = classify_ingredients(vision_fixture())
        by_name = {item["name"]: item["risk"] for item in items}

        self.assertEqual(by_name["氢化植物油"], "high")
        self.assertEqual(by_name["阿斯巴甜"], "medium")
        self.assertEqual(by_name["柠檬酸"], "safe")

    def test_missing_nutrition_uses_conservative_summary(self):
        data = {
            "product_name": "缺失营养表",
            "ingredients": [],
            "additives": [],
            "nutrition": {"per_100g": {}},
        }

        self.assertFalse(has_nutrition_data(data))
        summary = generate_summary(
            "low",
            calc_indicators(data),
            compute_benchmarks(data),
            nutrition_available=False,
        )
        self.assertIn("营养成分数据缺失", summary)

    def test_low_protein_summary_not_called_balanced(self):
        data = neutral_fixture(
            fat_g=6,
            sodium_mg=50,
            sat_fat_g=0,
            sugar_g=0,
            protein_g=0,
            trans_fat_g=0,
        )

        lights = calc_indicators(data)
        risk = calc_risk(data, profile=None)
        summary = generate_summary(risk, lights, compute_benchmarks(data))

        self.assertEqual(lights["protein"], "red")
        self.assertIn("蛋白质较低", summary)
        self.assertNotIn("营养均衡", summary)

    def test_unrelated_profile_concern_does_not_raise_risk(self):
        data = neutral_fixture(
            fat_g=6,
            sodium_mg=50,
            sat_fat_g=0,
            sugar_g=0,
            carb_g=20,
            protein_g=8,
            trans_fat_g=0,
        )

        self.assertEqual(calc_risk(data, profile={"concerns": ["坚果过敏"]}), "low")

    def test_matching_profile_concern_can_raise_risk(self):
        data = neutral_fixture(
            fat_g=6,
            sodium_mg=350,
            sat_fat_g=0,
            sugar_g=0,
            carb_g=20,
            protein_g=8,
            trans_fat_g=0,
        )

        self.assertEqual(calc_risk(data, profile=None), "low")
        self.assertEqual(calc_risk(data, profile={"concerns": ["高血压"]}), "medium")

    def test_profile_calculations(self):
        self.assertEqual(calc_bmi(65, 170), {"value": 22.5, "level": "正常"})

        tdee = calc_tdee("男", weight=65, height=170, age=30, activity="中度")
        self.assertEqual(tdee["bmr"], 1568)
        self.assertEqual(tdee["tdee"], 2430)


if __name__ == "__main__":
    unittest.main()
