FROM python:3.7-slim as builder
RUN apt-get update -y;
RUN apt-get upgrade -y;

RUN apt-get install -y unzip
RUN pip3 install --upgrade pip

RUN apt-get -y install nano git wget
RUN apt-get clean

RUN mkdir /efs

COPY SyntheticDataGenerator.zip .
RUN unzip SyntheticDataGenerator.zip

RUN python -m venv SyntheticDataGenerator
WORKDIR SyntheticDataGenerator

EXPOSE 8055

RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["run_server"]
ENTRYPOINT ["python", "app.py"]