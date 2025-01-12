#!/bin/bash

export SSH_AUTH_SOCK=/ssh-agent
ssh-keyscan -H $GLUEX_HOSTNAME >> /root/.ssh/known_hosts

uv pip install -e .

luigid --background --pidfile /var/run/luigi.pid --logdir /var/log/luigi &

sleep 2

echo "Luigi dashboard available at http://localhost:8082"

cd /work

if [ "$1" = "run-analysis" ]; then
    luigi --module thesis_analysis RunAll \
        --global-parameters-username="$GLUEX_USERNAME" \
        --global-parameters-hostname="$GLUEX_HOSTNAME" \
        --workers=16
    read -n 1 -s -r -p "Press any key to continue..."
    echo "\n"
    /bin/bash
else
    /bin/bash
fi
