FROM python:3.11-slim

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_DEFAULT_TIMEOUT=300
ENV HF_ENDPOINT=https://hf-mirror.com

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn && echo "torch OK" \
  && pip install --no-cache-dir fastapi uvicorn openai python-dotenv pydantic redis langchain langchain-openai && echo "web OK" \
  && pip install --no-cache-dir sentence-transformers chromadb numpy && echo "ml OK"

WORKDIR /app
COPY . .
VOLUME ["/app/chroma_db", "/app/sessions"]
EXPOSE 8000

CMD ["python", "-u", "server.py"]
