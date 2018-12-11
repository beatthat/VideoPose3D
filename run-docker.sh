#!/bin/bash
set -e
ROOT=`pwd`

NAME=fastai

# mkdir -p docker_host
HOST_MOUNT=`pwd`

docker run \
  --name ${NAME} \
  -it \
  --rm \
  --shm-size 16G \
  --runtime=nvidia \
  -p 8888:8888 \
  --mount type=bind,src=${HOST_MOUNT},target=/docker_host \
  --workdir /docker_host \
  videopose3d bash


# docker exec -it ${NAME} bash
