# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy everything into container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI app from `api/main.py`
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
