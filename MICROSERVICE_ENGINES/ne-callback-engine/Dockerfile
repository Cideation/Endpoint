FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir networkx
CMD ["python", "-c", "print('ne-callback-engine container ready')"] 