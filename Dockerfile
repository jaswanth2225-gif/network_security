FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create necessary directories
RUN mkdir -p final_model prediction_output Artifacts logs

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
