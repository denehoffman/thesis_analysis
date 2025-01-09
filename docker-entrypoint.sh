#!/bin/bash

luigid --background --pidfile /var/run/luigi.pid --logdir /var/log/luigi &

sleep 2

echo "Luigi dashboard available at http://localhost:8082"

cd /work

if [ "$1" = "run-analysis" ]; then
    luigi --module thesis_analysis RunAll \
        --global-parameters-username="$USERNAME" \
        --global-parameters-hostname="$HOSTNAME"
else
    /bin/bash
fi
