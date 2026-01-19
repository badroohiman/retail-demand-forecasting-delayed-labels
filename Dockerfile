FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1

# Example: run full pipeline
CMD ["python", "-m", "src.train", "--full", "--label_version", "y_v2", "--out_dir", "/tmp/outputs"]
