FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model_handler.py config.py ./

RUN mkdir -p /app/models /app/temp_results

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_OFFLINE=0

EXPOSE 7860

CMD ["python", "app.py"]

