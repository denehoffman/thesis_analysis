#!/bin/bash

export SSH_AUTH_SOCK=/ssh-agent
ssh-keyscan -H "$GLUEX_HOSTNAME" >>/root/.ssh/known_hosts

uv pip install -e .

luigid --background --pidfile /var/run/luigi.pid --logdir /var/log/luigi &

sleep 2

echo "Luigi dashboard available at http://localhost:8082"

cd /work

if [ "$1" = "run-analysis" ]; then
  echo "Starting analysis"
  luigi --module thesis_analysis RunAll --workers="$2"
  read -n 1 -s -r -p "Press any key to continue..."
  echo "\n"
  /bin/bash
else
  /bin/bash
fi
