FROM python:3.11-slim
WORKDIR /app
COPY . /app
COPY ../shared /app/shared
ENV PYTHONPATH="${PYTHONPATH}:/app/shared"
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "main.py"] 