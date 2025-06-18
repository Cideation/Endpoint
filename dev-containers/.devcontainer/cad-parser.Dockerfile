FROM python:3.11-slim
WORKDIR /app
COPY ./services/cad-parser /app
RUN pip install --no-cache-dir flask
CMD ["python", "app.py"] 