FROM python:3.10

WORKDIR /BA-ParticleTrackingSystem/ParticleTrackingSystem

COPY . .

COPY ./requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg
RUN pip install bokeh

COPY . .

EXPOSE 5006

RUN ["python3",  "launcher.py"]

COPY . .

WORKDIR ../
CMD  ["python", "-m", "bokeh", "serve", "--show", "./ParticleTrackingSystem"]