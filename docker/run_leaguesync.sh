#! /bin/bash

set -e

echo '============= LeagueSync ============'> /opt/leaguesync-data/log.txt
date >> /opt/leaguesync-data/log.txt

leaguesync -c /etc/leaguesync.env -vv sync  | tee -a /opt/leaguesync-data/log.txt

leaguesync  -c /etc/leaguesync.env -vv extract -d /opt/leaguesync-data | tee -a /opt/leaguesync-data/log.txt

date >> /opt/leaguesync-data/log.txt

