FROM pathwaycom/pathway:latest

# Set working directory
WORKDIR /app

# Copy only requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Ensure Python can find src/
ENV PYTHONPATH=/app

# Run the app
CMD ["python", "main.py"]
