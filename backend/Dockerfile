FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the index at image build time so the container starts ready to serve.
# (Re-run scripts/build_index.py --reset if you change backend/data/corpus.json.)
RUN python -m scripts.build_index --reset

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
