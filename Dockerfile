FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential

WORKDIR /app

# Download lid.176.bin into /app
RUN curl -L -o lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]