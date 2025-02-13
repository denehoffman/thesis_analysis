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

@run:
  docker run -it \
      -p 8082:8082 \
      -v "$(pwd):/work" \
      --mount type=bind,source=$SSH_AUTH_SOCK,target=/ssh-agent \
      --env SSH_AUTH_SOCK=/ssh-agent \
      -e GLUEX_USERNAME="nhoffman" \
      -e GLUEX_HOSTNAME="ernest.phys.cmu.edu" \
      pyroot-env run-analysis

@open:
  python3 -m webbrowser "localhost:8082"

@transfer:
  cp analysis/plots/* ../../thesis/figures/
