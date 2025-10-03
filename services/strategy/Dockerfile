FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir nats-py prometheus-client
CMD ["python", "app.py"]
