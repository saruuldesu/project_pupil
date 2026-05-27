FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git gcc g++ ninja-build build-essential \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir -r requirements.txt


COPY . .

RUN test -f main.py
RUN test -f make.py
RUN test -f models_pre/Config/config.yaml
RUN test -f models_pre/model_final.pth

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port $PORT"]