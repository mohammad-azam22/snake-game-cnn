# Base Image
FROM python:3.12.12-slim

WORKDIR /app

COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt

COPY ["app.py", "inference.py", "model.py", "preprocessing.py", "train.py", "./"]

COPY ["./models/letter_cnn_v1.keras", "./models/"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--host=0.0.0.0", "--port=9696", "app:app"]
