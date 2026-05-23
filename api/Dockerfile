FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY api/ ./api/
COPY app/config.py ./app/config.py

# Copy data files
COPY outputs/ ./outputs/
COPY data/processed/player_countries.csv ./data/processed/player_countries.csv

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
