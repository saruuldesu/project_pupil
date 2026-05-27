FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git gcc g++ ninja-build build-essential \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN python -m pip install "setuptools<70" wheel
RUN python -m pip install --no-cache-dir -r requirements.txt

ENV FORCE_CUDA=0
ENV MAX_JOBS=1

RUN git clone https://github.com/facebookresearch/detectron2.git /tmp/detectron2
RUN cd /tmp/detectron2 && \
    python -m pip install -e . --no-build-isolation --no-cache-dir

RUN python -c "from detectron2.config import get_cfg; print('detectron2 ok')"
RUN python -c "import sys; print(sys.executable)"
RUN python -c "import detectron2; print(detectron2.__file__)"
RUN python -c "from detectron2.config import get_cfg; print('OK')"
COPY . .

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port $PORT"]