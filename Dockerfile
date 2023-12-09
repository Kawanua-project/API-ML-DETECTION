FROM python:3.9

ENV PORT 8080
ENV HOST 0.0.0.0

EXPOSE 8080

RUN apt-get update -y && \
    apt-get install -y python3-pip

COPY ./requirement.txt /app/requirement.txt

WORKDIR /app

RUN pip install -r requirement.txt

COPY . /app

ENTRYPOINT ["python", "app.py"]