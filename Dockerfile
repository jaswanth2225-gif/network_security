FROM python:3.10-slim-bullseye

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install awscli

# Copy the entire application
COPY . /app

# Create necessary directories
RUN mkdir -p final_model prediction_output Artifacts logs

# Expose port
EXPOSE 8080

# Run the application
CMD ["python3", "app.py"]
