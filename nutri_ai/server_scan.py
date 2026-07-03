"""
食鉴 2.0 — 扫描后端
POST /vision   上传配料表照片 → {product_name, ingredients, nutrition, additives}
POST /analyze  OCR结果+画像 → {risk, suggestion, ingredients, advice}
GET  /        前端页面
"""
import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vision import scan_label
from analyze import full_analysis

app = FastAPI(title="食鉴2.0")


@app.post("/vision")
async def vision(image: UploadFile = File(...)):
    suffix = Path(image.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name
    try:
        data = scan_label(tmp_path)
        return data
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class AnalyzeRequest(BaseModel):
    vision_json: dict
    profile: dict | None = None


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    result = full_analysis(req.vision_json, req.profile)
    return result


static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("🍎 食鉴 2.0 — http://127.0.0.1:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
