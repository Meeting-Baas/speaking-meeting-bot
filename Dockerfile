FROM python:3.11-slim

WORKDIR /app

# Install system dependencies 
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not use virtualenvs inside Docker
RUN poetry config virtualenvs.create false

# Regenerate lock file and install dependencies
RUN poetry lock && poetry install --no-interaction --no-ansi --no-root

# Copy application files
COPY . .

# Set Python path to include the current directory
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Compile protocol buffers
RUN poetry run python -m grpc_tools.protoc --proto_path=./protobufs --python_out=./protobufs frames.proto

# Environment variables
ENV PORT=7014

EXPOSE ${PORT}

CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT}"]

