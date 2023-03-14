# use ubuntu 20.04 because we want to use ROS noetic
ARG TARGET="gpu"

FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

#ARG CUDA_VERSION="11.3.1"
ARG nvidia_driver_major_version=525
ARG nvidia_binary_version="${nvidia_driver_major_version}.161.03"

#FROM nvidia_driver:11.3.1-base-ubuntu20.04
# FROM ros:noetic
LABEL maintainer "Gašper Šavle <gaspersavle13@gmail.com>"

SHELL ["/bin/bash","-c"]
################################################################
## BEGIN: ROS:core
################################################################

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install packages
RUN apt update && apt install -q -y --no-install-recommends \
    git \
    software-properties-common \
    build-essential \
    apt-utils \
    wget \
    libgl1-mesa-glx \
    dirmngr \
    gnupg2 \
    curl \
    nano \
    libglib2.0-0

RUN apt update
# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO noetic

################################################################
## END: ROS:core
## BEGIN: Python 3
################################################################

RUN add-apt-repository universe

RUN apt update --fix-missing && apt install -y wget bzip2 ca-certificates zlib1g-dev libncurses5-dev libgdbm-dev \
    libglib2.0-0 libxext6 libsm6 libxrender1 libffi-dev \
    libusb-1.0-0-dev

RUN apt update && apt install -y python3-pip
RUN pip3 install --upgrade pip

RUN pip3 install empy jupyter quaternionic shapely nano pycryptodomex install setuptools gnupg requests
RUN pip3 install -U rospkg six


################################################################
## END: Python 3
## BEGIN: realsense
################################################################

RUN apt install -y apt-transport-https
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE \
    && add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u

################################################################
## END: realsense
## BEGIN: ROS
################################################################

# Create local catkin workspace
ENV CATKIN_WS=/root/catkin_ws
ENV ROS_PYTHON_VERSION=3
RUN mkdir -p $CATKIN_WS/src/nn_pipeline
WORKDIR $CATKIN_WS

COPY nn_pipeline $CATKIN_WS/src/nn_pipeline


RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt update
RUN apt install -y ros-noetic-ros-base \
                   python3-rospy \
                   python3-catkin-tools \
                   python3-osrf-pycommon
RUN source /opt/ros/${ROS_DISTRO}/setup.bash
RUN apt install -y ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-vision-opencv

RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt update
RUN apt install -y ros-noetic-ros-base \
                   python3-rospy 
RUN source /opt/ros/${ROS_DISTRO}/setup.bash


# Always source ros_catkin_entrypoint.sh when launching bash (e.g. when attaching to container)
RUN echo "source /entrypoint.sh" >> /root/.bashrc

###############################################################
## END: ROS
## BEGIN: AlphaPose
################################################################
RUN apt-get install python3-tk -y

RUN apt update && apt install -y python3-pip
RUN pip3 install --upgrade pip

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip3 install rich pillow scikit-learn scipy tensorboard pyyaml opencv-python regex natsort shapely commentjson pycocotools scikit-image joblib==1.1.0 jsonpickle
RUN pip3 install lap \
    numpy==1.19.5 \
    matplotlib==3.6 \
    Cython==0.29.21 \
    requests 
RUN pip3 install cython_bbox
RUN apt update

WORKDIR /
RUN git clone https://github.com/MVIG-SJTU/AlphaPose.git
WORKDIR /AlphaPose

ENV PATH=/usr/local/cuda/bin/:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
RUN pip install cython
RUN apt-get install libyaml-dev -y
RUN python3 setup.py build develop --user

# Dodajanje utezi natreniranega modela
WORKDIR /nn_pipeline/AlphaPose/detector/yolo
RUN mkdir data

# Dodajanje negotovosti modela
WORKDIR /nn_pipeline/AlphaPose/detector/tracker
RUN mkdir data

WORKDIR /nn_pipeline/AlphaPose
RUN mkdir output

COPY pret_model $CATKIN_WS/src/nn_pipeline/AlphaPose/pretrained_models
COPY test_model $CATKIN_WS/src/n_pipeline/AlphaPose/detector/tracker/data
COPY test_weights $CATKIN_WS/src/nn_pipeline/AlphaPose/detector/yolo/data

###############################################################
## END: AlphaPose
################################################################
WORKDIR /usr/lib
RUN rm -r python3 \
          python2.7 \
          python3.9
         # python3.8

WORKDIR /

WORKDIR /
RUN wget https://raw.githubusercontent.com/BorisKuster/AllNet/main/entrypoint.sh
#COPY ./entrypoint.sh /
RUN chmod +x entrypoint.sh

RUN wget https://raw.githubusercontent.com/BorisKuster/AllNet/main/run-jupyter
#COPY ./run-jupyter /bin/run-jupyter
RUN chmod +x run-jupyter

ENTRYPOINT ["/entrypoint.sh"]

# # stop docker from exiting immediately
CMD tail -f /dev/null

