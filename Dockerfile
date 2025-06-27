FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
