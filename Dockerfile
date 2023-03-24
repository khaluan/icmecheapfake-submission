FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

# Setup Environment Variables
ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Asia/Vietnam"

WORKDIR /icmecheapfakes-src/

# Copy Dependencies
COPY . ./

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

RUN pip3 install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download>


ENTRYPOINT []
CMD [ "python3", "main.py" ]