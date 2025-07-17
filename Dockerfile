FROM python:3.10-slim

# Install system dependencies for fasttext
RUN apt-get update && apt-get install -y curl build-essential \
    && curl -L -o lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Set work directory
WORKDIR /app

# Copy requirements (create requirements.txt with all your dependencies)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project files
COPY . .

# Expose FastAPI port
EXPOSE 8080

# Start FastAPI app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

