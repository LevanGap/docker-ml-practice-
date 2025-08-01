# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Run the Flask app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
