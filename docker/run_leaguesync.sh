#! /bin/bash

set -e

leaguesync -c /etc/leaguesync.env -vv sync 

leaguesync  -c /etc/leaguesync.env -vv extract -d /opt/leaguesync-data