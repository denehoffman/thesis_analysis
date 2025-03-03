@default:
  just --list

@build:
  docker build -t pyroot-env .

@shell:
  docker run -it \
      -p 8082:8082 \
      -v "$(pwd):/work" \
      --mount type=bind,source=$SSH_AUTH_SOCK,target=/ssh-agent \
      --env SSH_AUTH_SOCK=/ssh-agent \
      -e GLUEX_USERNAME="nhoffman" \
      -e GLUEX_HOSTNAME="ernest.phys.cmu.edu" \
      pyroot-env

@shell-mac:
  docker run -it \
      -p 8082:8082 \
      -v "$(pwd):/work" \
      --mount type=bind,source=/run/host-services/ssh-auth.sock,target=/ssh-agent \
      --env SSH_AUTH_SOCK=/ssh-agent \
      -e GLUEX_USERNAME="nhoffman" \
      -e GLUEX_HOSTNAME="ernest.phys.cmu.edu" \
      pyroot-env

@run:
  docker run -it \
      -p 8082:8082 \
      -v "$(pwd):/work" \
      --mount type=bind,source=$SSH_AUTH_SOCK,target=/ssh-agent \
      --env SSH_AUTH_SOCK=/ssh-agent \
      -e GLUEX_USERNAME="nhoffman" \
      -e GLUEX_HOSTNAME="ernest.phys.cmu.edu" \
      pyroot-env run-analysis 16

@run-mac:
  docker run -it \
      -p 8082:8082 \
      -v "$(pwd):/work" \
      --mount type=bind,source=/run/host-services/ssh-auth.sock,target=/ssh-agent \
      -e SSH_AUTH_SOCK=/ssh-agent \
      -e GLUEX_USERNAME="nhoffman" \
      -e GLUEX_HOSTNAME="ernest.phys.cmu.edu" \
      pyroot-env run-analysis 4


#@run-local WORKERS
#    #!/usr/bin/env bash
#    set -euxo pipefail
#    [ -d .venv ] || uv venv
#    uv pip install -e .
#    luigid --background --pidfile ./luigi.pid --#logdir ./luigi.log &
#    luigi --module thesis_analysis RunAll \
#        --global-parameters-username="nhoffman" \
#        --global-parameters-hostname="ernest.#phys.cmu.edu" \
#        --workers="{{WORKERS}}"

@open:
  python3 -m webbrowser "localhost:8082"

@transfer:
  cp analysis/plots/* ../../thesis/figures/
