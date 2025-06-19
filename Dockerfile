FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y gcc build-essential

# Set working directory
WORKDIR /app

# Copy pip requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose the port your app listens on (adjust if needed)
EXPOSE 5000

# Install gunicorn and run the app
RUN pip install gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.index:app"]