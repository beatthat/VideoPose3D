FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install \
  -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
     build-essential \
     ca-certificates \
     cmake \
     curl \
     git \
     libjpeg-dev \
     libnccl2=2.2.13-1+cuda9.0 \
     libnccl-dev=2.2.13-1+cuda9.0 \
     libpng-dev \
     python-qt4 \
  	 unzip \
     vim \
     wget \
  	 zip && \
   rm -rf /var/lib/apt/lists/*


ENV PYTHON_VERSION=3.7
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
    /opt/conda/bin/conda install conda-build


WORKDIR /notebooks

# clone fastai and also download/install its weights for its pretrained models
RUN git clone https://github.com/fastai/fastai.git .
RUN curl http://files.fast.ai/models/weights.tgz --output fastai/weights.tgz
RUN tar xvfz fastai/weights.tgz -C fastai
RUN ln -s /notebooks/fastai/weights /notebooks/old/fastai/weights

RUN ls && /opt/conda/bin/conda env create
RUN /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/fastai/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV USER fastai

# CMD source activate fastai
# CMD source ~/.bashrc

#RUN /opt/conda/bin/conda install jupyterlab

WORKDIR /data

RUN curl http://files.fast.ai/data/dogscats.zip --output dogscats.zip
RUN unzip -d . dogscats.zip
RUN rm dogscats.zip

RUN ln -s /data/ /notebooks/courses/dl1/
RUN ls -la /notebooks/courses/dl1/data/

RUN chmod -R a+w /notebooks

ENV PATH /opt/conda/bin:$PATH
WORKDIR /notebooks

ENV PATH /opt/conda/envs/fastai/bin:$PATH

# RUN pip install numpy torchvision_nightly
# RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html
RUN pip install http://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision

RUN pip install fastai

# jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
# CMD ["source", "activate", "fastai"]
