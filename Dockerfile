FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py .
COPY best_model.pth .

# Expose port
EXPOSE 7860

# Set environment variables
ENV FLASK_APP=api.py
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "-c", "from api import app; app.run(host='0.0.0.0', port=7860)"]