# Use Ollama base image
FROM ollama/ollama:latest

# Set working directory
WORKDIR /app

# Install Python and pip
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip3 install --no-cache-dir langchain langchain-community rank_bm25 langchain_ollama --break-system-packages

COPY . /app

# Download llama3.2 3B model
RUN ollama serve & \
    sleep 5 && \
    ollama pull llama3.2:3b && \
    ollama pull bge-m3

# Generate retriever and vectorstore at build time
RUN ollama serve & \
    sleep 5 && \
    python3 vectorize_documents.py

# Simple initialization script
CMD ["echo", "Hello from ollama"]
