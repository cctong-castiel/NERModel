FROM python:3.8.2

WORKDIR /app

ADD . /app

RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install git+https://www.github.com/keras-team/keras-contrib.git
RUN python3 -m pip install -r requirements.txt

EXPOSE 721

CMD gunicorn -c ./gunicorn.conf main:app
