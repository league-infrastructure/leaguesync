# Many ideas from  https://blog.thesparktree.com/cron-in-docker


# Use a lightweight Python base image
FROM python:3.11-slim

# Install git
RUN apt-get update &&\
 apt-get install -y procps build-essential git cron &&\
 which cron &&\
 rm -rf /etc/cron.*/* &&\
 apt-get clean 


COPY requirements.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt

RUN pip install git+https://github.com/league-infrastructure/leaguesync.git


COPY leaguesync.env /etc/leaguesync.env
COPY run_leaguesync.sh  /opt/run_leaguesync.sh

COPY crontab /etc/crontab # Cron wont run w/o this


CMD ["cron","-f", "-l", "15"]