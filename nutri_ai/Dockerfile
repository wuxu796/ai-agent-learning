FROM python:3.14-slim

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_DEFAULT_TIMEOUT=300
ENV HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

# 先装 torch（CPU 版，体积小），再装其余依赖
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && echo "torch OK"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && echo "deps OK"

COPY . .

VOLUME ["/app/chroma_db", "/app/sessions"]
EXPOSE 8000

CMD ["python", "-u", "server.py"]
