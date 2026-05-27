FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git gcc g++ ninja-build build-essential \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN python -m pip install "setuptools<70" wheel

# Install app deps first
RUN python -m pip install --no-cache-dir -r requirements.txt

# Install Detectron2 from source
RUN git clone https://github.com/facebookresearch/detectron2.git /tmp/detectron2
RUN cd /tmp/detectron2 && \
    FORCE_CUDA=0 MAX_JOBS=1 \
    python -m pip install -e . --no-build-isolation --no-cache-dir

# Verify Detectron2 before copying app
RUN python -m pip show detectron2
RUN python -c "from detectron2.config import get_cfg; print('detectron2 ok')"

COPY . .

RUN test -f main.py
RUN test -f make.py
RUN test -f models_pre/Config/config.yaml
RUN test -f models_pre/model_final.pth

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port $PORT"]