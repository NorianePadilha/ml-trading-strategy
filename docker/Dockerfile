FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

COPY src/ src/
COPY api/ api/
COPY feature_store/ feature_store/
COPY monitoring/ monitoring/
COPY models/ models/
COPY results/ results/
COPY logs/ logs/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
