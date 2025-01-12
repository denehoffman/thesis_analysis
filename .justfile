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


clean level:
  #!/usr/bin/env bash
  set -euxo pipefail
  case {{level}} in
    "accpol")
      find analysis/datasets -type d -name "accpol" -print -exec rm -rf {} \;
      ;;
    "chisqdof")
      find analysis/datasets -type d -name "chisqdof_*" -print  -exec rm -rf {} \;
      ;;
    "splot")
      find analysis/datasets -type d -name "splot*" -print -exec rm -rf {} \;
      ;;
    "plots")
      rm -rf analysis/plots/*
      ;;
    *)
      echo "Unknown level: {{level}}. Valid levels are: accpol, chisqdof, splot."
      ;;
  esac
