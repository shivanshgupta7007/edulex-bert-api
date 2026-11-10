FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
# To serve the trained checkpoint, also: COPY model/ model/
# and install torch + transformers (see scripts/ADD_MODEL_HERE.md)

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
