FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel 

RUN apt-get update && apt-get install \
    -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
    vim \
    wget \
    unzip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
  h5py \
  jupyter 

RUN git clone https://github.com/facebookresearch/VideoPose3D.git /videopose3d

# download pretrained model for detectron to checkpoints
WORKDIR /videopose3d/checkpoint
RUN wget https://s3.amazonaws.com/video-pose-3d/d-pt-243.bin

# WORKDIR /videopose3d/data
# RUN wget https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip && \
#   python prepare_data_h36m.py --from-archive h36m.zip


ENV PYTHONPATH=/videopose3d:PYTHONPATH
WORKDIR /videopose3d