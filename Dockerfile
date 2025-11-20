# ---- builder (optional for caching) ----
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt requirements-api.txt ./
RUN pip install --upgrade pip && \
    pip wheel --wheel-dir=/wheels -r requirements.txt -r requirements-api.txt && \
    cp requirements.txt requirements-api.txt /wheels/

# ---- runtime ----
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# system deps if needed (uncomment if pdf libs required)
# RUN apt-get update && apt-get install -y libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels -r /wheels/requirements.txt -r /wheels/requirements-api.txt

# copy app
COPY . /app
ENV DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
