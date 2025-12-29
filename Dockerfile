FROM python:3.10-slim-buster

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN apt update -y && apt install awscli -y

RUN apt-get update && pip install -r requirements.txt

# Copy the entire application
COPY . /app

# Create necessary directories
RUN mkdir -p final_model prediction_output Artifacts logs

# Expose port
EXPOSE 8080

# Run the application
CMD ["python3", "app.py"]
