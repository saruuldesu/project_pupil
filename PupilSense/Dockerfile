FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git gcc g++ ninja-build build-essential \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV FORCE_CUDA=0
ENV MAX_JOBS=1

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN python -m pip install "setuptools<70" wheel
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN python -m pip install --no-cache-dir \
    torch==2.8.0 torchvision==0.23.0 torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

RUN python -m pip install --no-cache-dir --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git'

COPY . .

RUN python -c "from detectron2.config import get_cfg; print('detectron2 ok')"

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]