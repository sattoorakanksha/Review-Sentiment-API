FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --default-timeout=300 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 -r requirements.txt

COPY app.py .
COPY model_output/final_model ./model_output/final_model

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]