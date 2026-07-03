# Use Ollama base image
FROM ollama/ollama:latest

# Set working directory
WORKDIR /app

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip3 install --no-cache-dir langchain langchain_community rank_bm25

# Copy your application code if needed
# COPY . /app

# Download llama3.2 3B model
RUN ollama pull llama3.2:3b

# Generate retriever and vectorstore at build time
RUN python3 vectorize_documents.py

# Simple initialization script
CMD ["echo", "Hello from ollama"]
