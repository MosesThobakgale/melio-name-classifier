FROM --platform=linux/amd64 python:3.9.18-slim

WORKDIR /app

# Dependencies
COPY ./deployment/serve-requirements.txt .
RUN pip install --no-cache-dir  -r serve-requirements.txt

# Make custom src code visible

ENV PYTHONPATH "${PYTHONPATH}:/app"

# Trained model and definition with main script
COPY ./saved_model /app/saved_model
COPY ./deployment/main.py /app/main.py

# Set entrypoint
ENTRYPOINT ["python", "-m", "main"]
