FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY demo.py ./
COPY README.md ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

CMD ["python", "demo.py"]

