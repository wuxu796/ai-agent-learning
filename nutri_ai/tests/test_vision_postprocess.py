"""视觉识别后处理的稳定测试。

这些测试不调用真实视觉模型，只保护模型输出 JSON 的容错解析和后处理。
"""

import unittest

try:
    from vision import _extract_json, _fix_per_serving, _split_ingredients, _validate_schema
except ModuleNotFoundError as e:
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"vision 依赖未安装: {_IMPORT_ERROR}")
class VisionPostprocessTest(unittest.TestCase):
    def test_extract_json_from_markdown_and_extra_text(self):
        text = """好的，识别结果如下：
```json
{"product_name":"测试食品","ingredients":["小麦粉"],"nutrition":{"per_100g":{"energy_kj":"1200"}},"additives":[]}
```
"""

        data = _extract_json(text)

        self.assertEqual(data["product_name"], "测试食品")
        self.assertEqual(data["ingredients"], ["小麦粉"])

    def test_validate_schema_coerces_numeric_strings(self):
        data = _validate_schema({
            "product_name": "测试食品",
            "ingredients": ["小麦粉"],
            "nutrition": {
                "per_100g": {
                    "energy_kj": "1200",
                    "fat_g": "8.5",
                    "sodium_mg": None,
                }
            },
            "additives": [],
        })

        per100g = data["nutrition"]["per_100g"]
        self.assertEqual(per100g["energy_kj"], 1200.0)
        self.assertEqual(per100g["fat_g"], 8.5)
        self.assertIsNone(per100g["sodium_mg"])

    def test_split_ingredients_and_additives(self):
        data = {
            "ingredients": ["小麦粉、白砂糖、植物油（含大豆）"],
            "additives": ["食品添加剂（山梨酸钾、柠檬酸）"],
        }

        _split_ingredients(data)

        self.assertEqual(data["ingredients"], ["小麦粉", "白砂糖", "植物油"])
        self.assertEqual(data["additives"], ["山梨酸钾", "柠檬酸"])

    def test_fix_per_serving_converts_small_values_to_per_100g(self):
        data = {
            "nutrition": {
                "serving_g": 25,
                "per_100g": {
                    "energy_kj": 500,
                    "fat_g": 4,
                    "sodium_mg": 100,
                },
            }
        }

        _fix_per_serving(data)

        per100g = data["nutrition"]["per_100g"]
        self.assertEqual(per100g["energy_kj"], 2000)
        self.assertEqual(per100g["fat_g"], 16)
        self.assertEqual(per100g["sodium_mg"], 400)


if __name__ == "__main__":
    unittest.main()
