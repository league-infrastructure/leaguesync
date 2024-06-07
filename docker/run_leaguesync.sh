#! /bin/bash

set -e

leaguesync -c /etc/league-remote.env -vv sync 

leaguesync -vv extract -d /opt/leaguesync-data