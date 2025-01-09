@default:
  just --list

@build:
  docker build -t pyroot-env .

@shell: build
  if [ -d "$(pwd)/analysis" ]; then \
    docker run -it \
        -p 8082:8082 \
        -v "$(pwd)/analysis:/work/analysis" \
        pyroot-env; \
  else \
    docker run -it \
        -p 8082:8082 \
        pyroot-env; \
  fi

@run: build
  if [ -d "$(pwd)/analysis" ]; then \
    docker run -it \
        -p 8082:8082 \
        -v "$(pwd)/analysis:/work/analysis" \
        -e USERNAME="nhoffman" \
        -e HOSTNAME="ernest.phys.cmu.edu" \
        pyroot-env run-analysis; \
  else \
    docker run -it \
        -p 8082:8082 \
        -e USERNAME="nhoffman" \
        -e HOSTNAME="ernest.phys.cmu.edu" \
        pyroot-env run-analysis; \
  fi

@open:
  python3 -m webbrowser "localhost:8082"
