FROM python:3.11-slim

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y gcc build-essential

# Set working directory
WORKDIR /app

# Copy pip requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose the port your app listens on (adjust if needed)
EXPOSE 5000

# Install gunicorn
RUN pip install gunicorn

# Start the app using gunicorn; adjust the module path if needed
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.index:app"]