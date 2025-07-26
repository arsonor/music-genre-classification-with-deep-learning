FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir pandas flask evidently==0.7.11 pyarrow

CMD ["python", "run_monitoring.py"]