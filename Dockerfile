# FROM python:3.7.13
# FROM python:3.7.16-slim
# FROM nvidia/cuda:11.4.0-base-ubuntu20.04
FROM nvidia/cuda:11.2.2-base-ubuntu20.04
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RUNNING_IN_DOCKER="True"

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install g++ -y

RUN apt-get install -y python3.9
RUN apt-get -y install python3.9-distutils
RUN apt-get install -y python3-pip

# Update Python version
# RUN apt-get update && apt-get install -y python3.9 python3.9-distutils

# Update pip to use the correct version of Python
# RUN python3.9 -m ensurepip --upgrade
# RUN python3.9 -m pip install --upgrade pip


RUN pip install IPython
RUN pip install shapely==2.0.4
RUN pip install networkx==3.1
RUN pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip install ultralytics==8.3.48


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglu1-mesa \
    libgl1 \
    libffi-dev \
    libglib2.0-0

WORKDIR /workspace
COPY requirements.txt /workspace
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace

# Define the APP_VERSION variable
ARG APP_VERSION=1.0.0
ENV APP_VERSION=${APP_VERSION}

CMD ["python3", "services/code/main.py"]
