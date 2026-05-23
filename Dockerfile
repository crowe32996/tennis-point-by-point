FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code (includes player_countries.csv now)
COPY api/ ./api/

# Copy config
RUN mkdir -p ./app
COPY app/config.py ./app/config.py

# Copy DuckDB database
COPY outputs/ ./outputs/

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
