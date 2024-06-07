# Use a lightweight Python base image
FROM python:3.10-slim

# Install git
RUN apt-get update && apt-get install -y git cron

# Copy the environment file to /etc
COPY league-remote.env /etc/league-remote.env

# Copy the environment file to /etc
COPY requirements.txt /opt/requirements.txt

# Add the cron job
RUN echo "*/30 * * * * root leaguesync -c /etc/league-remote.env -vv sync && leaguesync -vv extract -d /opt/leaguesync-data" >> /etc/cron.d/leaguesync-cron

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/leaguesync-cron

# Apply cron job
RUN crontab /etc/cron.d/leaguesync-cron

# Create the log file to be able to run tail
RUN touch /var/log/cron.log

RUN pip install -r /opt/requirements.txt

# Install the leaguesync module from GitHub
RUN pip install git+https://github.com/league-infrastructure/leaguesync.git #2

# Run the command on container startup
CMD cron && tail -f /var/log/cron.log
