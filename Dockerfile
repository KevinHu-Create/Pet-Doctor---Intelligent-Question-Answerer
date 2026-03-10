FROM  python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
#CMD ["python", "app.py"]
# 不能直接 python app.py，因为 app.py 只定义了 FastAPI 应用，不会启动服务
# 这里使用 uvicorn 启动 app.py 中的 app 对象，并监听 8000 端口
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]