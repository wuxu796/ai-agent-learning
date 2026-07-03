"""
PaddleOCR 包装器 — 输入图片，输出提取的文字。
由 vision.py 通过 subprocess 调用（使用 Python 3.12 + PaddleOCR）。

预处理管线：放大 → 去反光 → 增强对比 → OCR
"""
import json
import logging
import os
import sys
import warnings

import cv2
import numpy as np

os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_use_mkldnn"] = "0"
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="ch", show_log=False)

TARGET_WIDTH = 2500


def preprocess(image_path: str) -> str:
    """
    预处理：放大 + 灰度 + CLAHE 去反光。
    CLAHE 对包装袋反光场景有帮助，不能去掉。
    """
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    h, w = img.shape[:2]

    if w < TARGET_WIDTH:
        ratio = TARGET_WIDTH / w
        img = cv2.resize(img, (TARGET_WIDTH, int(h * ratio)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    out_path = os.path.splitext(image_path)[0] + "_prep.jpg"
    cv2.imwrite(out_path, gray)
    return out_path


def ocr_image(image_path: str) -> str:
    """OCR 识别，预处理增强后识别"""
    # 预处理
    prep_path = preprocess(image_path)

    result = ocr.ocr(prep_path, cls=True)
    if not result or not result[0]:
        return ""

    lines = []
    for line in result[0]:
        text = line[1][0]
        lines.append(text)
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("{}")
        sys.exit(1)

    image_path = sys.argv[1]
    text = ocr_image(image_path)
    print(json.dumps({"text": text}, ensure_ascii=False))
