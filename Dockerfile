FROM python:3.12.3-slim-bookworm

RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-dev \
    libglib2.0-0 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    tesseract-ocr \
    tesseract-ocr-jpn \
    libtesseract-dev \
    libleptonica-dev \
    tesseract-ocr-script-jpan \
    tesseract-ocr-script-jpan-vert \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /workspace
COPY pyproject.toml poetry.lock* ./
RUN poetry install
COPY . .