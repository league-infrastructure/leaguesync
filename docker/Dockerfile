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

RUN pip install git+https://github.com/league-infrastructure/leaguesync.git #3


COPY league-remote.env /etc/league-remote.env
COPY run_leaguesync.sh  /opt/run_leaguesync.sh
COPY crontab /etc/crontab
COPY entrypoint.sh /opt/entrypoint.sh

RUN crontab /etc/crontab

CMD ["cron","-f", "-l", "15"]