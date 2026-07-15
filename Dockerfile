# Use Ollama base image
FROM ollama/ollama:latest

# Install Python and pip
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip3 install --no-cache-dir flask flask-pydantic pydantic langchain langchain-community rank_bm25 langchain_ollama pylatexenc --break-system-packages

# Download llama3.2 3B model
RUN ollama serve & \
    sleep 5 && \
    ollama pull llama3.2:3b && \
    ollama pull bge-m3

# Set working directory
WORKDIR /app

COPY . /app
COPY tex/*.tex /app/tex/

RUN ollama serve & \
    sleep 5 && \
    python3 vectorize_documents.py

EXPOSE 5000

# Start Ollama server in the background and run server.py using the flask CLI
ENTRYPOINT ["/bin/bash", "-c", "ollama serve & sleep 5 && flask --app server run --host=0.0.0.0 --port=5000"]
