#!/bin/bash
set -e
ROOT=`pwd`

NAME=videopose3d

# mkdir -p docker_host
HOST_MOUNT=`pwd`

docker run \
  --name ${NAME} \
  -it \
  --rm \
  --shm-size 16G \
  --runtime=nvidia \
  -p 8888:8888 \
  -v=${HOST_MOUNT}:/docker_host \
  --workdir /docker_host \
  videopose3d jupyter notebook --allow-root --ip=0.0.0.0


# docker exec -it ${NAME} bash
