FROM python:3.9

ENV PORT 5000
ENV HOST 0.0.0.0

EXPOSE 5000

RUN apt-get update -y && \
    apt-get install -y python3-pip

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT ["python", "app.py"]