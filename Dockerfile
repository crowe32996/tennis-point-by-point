FROM python:3.11-slim

# Cache bust: v2
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

# Expose port (Railway uses PORT env var)
EXPOSE 8000

# Run the API - use shell form to expand $PORT
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
