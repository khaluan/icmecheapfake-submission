FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

# Setup Environment Variables
ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Asia/Vietnam"

WORKDIR /icmecheapfakes-src

# Copy Dependencies
COPY . ./

# COPY models/ ~/.cache/huggingface/hub/models--sshleifer--distilbart-cnn-12-6/

# Install Python
RUN apt-get update && \
    apt install -y software-properties-common  && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 python3-dev python3-pip python3-opencv \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

RUN pwd

# Prepare Python Dependencies
RUN python3 -m pip install --upgrade pip==21.2.2 && \
    pip3 install -r requirements.txt

RUN pip3 install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install -e detectron2
RUN pip3 install google-cloud-vision
RUN pip3 install protobuf==3.19.6
# RUN pip3 install --no-deps google-cloud-vision
RUN pip3 install https://github.com/amazon-science/ReFinED/archive/refs/tags/V1.zip
RUN pip3 install https://huggingface.co/spacy/en_core_web_trf/resolve/main/en_core_web_trf-any-py3-none-any.whl
RUN pip3 install https://github.com/explosion/spacy-experimental/releases/download/v0.6.1/en_coreference_web_trf-3.4.0a2-py3-none-any.whl

RUN pip3 install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# RUN mkdir models && cd cache && gdown 12kNONo0jgktxU0vWtV3Z2ZrCrB3DJPVj

ENTRYPOINT []
CMD [ "python3", "main.py" ]
