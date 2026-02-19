###############################################################################
# Dockerfile – ROS 2 Foxy development environment for SAFE_ROS2
# Base: ros:foxy (Ubuntu 20.04 / Python 3.8)
###############################################################################
FROM ros:foxy

ENV DEBIAN_FRONTEND=noninteractive

# ---------- ROS 2 apt dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    git \
    wget \
    # ROS 2 packages
    ros-foxy-rqt \
    ros-foxy-rqt-gui \
    ros-foxy-rqt-gui-py \
    ros-foxy-rqt-graph \
    ros-foxy-rqt-plot \
    ros-foxy-rqt-reconfigure \
    ros-foxy-rqt-service-caller \
    ros-foxy-rqt-topic \
    ros-foxy-rviz2 \
    ros-foxy-nav-msgs \
    ros-foxy-geometry-msgs \
    ros-foxy-std-msgs \
    ros-foxy-std-srvs \
    ros-foxy-visualization-msgs \
    ros-foxy-message-filters \
    ros-foxy-launch \
    ros-foxy-launch-ros \
    ros-foxy-rcl-interfaces \
    ros-foxy-ament-cmake \
    ros-foxy-ament-cmake-python \
    ros-foxy-rosidl-default-generators \
    ros-foxy-rosidl-default-runtime \
    ros-foxy-builtin-interfaces \
    # Lanelet2
    ros-foxy-lanelet2 \
    # X11 / GUI support
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libxcb-xinerama0 \
    x11-apps \
    python3-tk \
    gfortran \
    ffmpeg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# ---------- Newer Mesa drivers for modern GPUs (AMD/Intel) ----------
RUN add-apt-repository -y ppa:kisak/kisak-mesa && \
    apt-get update && apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

# ---------- Upgrade pip ----------
# Pin setuptools < 71 to avoid canonicalize_version incompatibility with pyspline
RUN python3 -m pip install --upgrade pip "setuptools<71" wheel

# ---------- Python pip dependencies ----------
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# ---------- JAX (CPU) for Python 3.8 ----------
# jaxlib wheels are hosted on Google storage, not PyPI
RUN python3 -m pip install --no-cache-dir jax==0.4.13 jaxlib==0.4.13 \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html

# ---------- hppfcl (collision checking) – build from source ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeigen3-dev libboost-dev libboost-serialization-dev \
    libassimp-dev liboctomap-dev python3-dev && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir eigenpy
RUN python3 -m pip install --no-cache-dir hpp-fcl

# ---------- Remove cmeel .pth that interferes with Python startup ----------
# eigenpy/hpp-fcl install via cmeel, whose .pth file runs during Python site
# init. Remove it and export the cmeel prefix path via PYTHONPATH instead.
RUN rm -f /usr/local/lib/python3.8/dist-packages/cmeel.pth
ENV PYTHONPATH="/usr/local/lib/python3.8/dist-packages/cmeel.prefix/lib/python3.8/site-packages:${PYTHONPATH}"

# ---------- Build pyspline from source (requires gfortran + numpy headers) ----------
RUN git clone --branch v1.5.2 --depth 1 https://github.com/mdolab/pyspline.git /tmp/pyspline && \
    cd /tmp/pyspline && \
    cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk && \
    sed -i 's/PYTHON = python$/PYTHON = python3/' config/config.mk && \
    make && \
    python3 -m pip install . && \
    rm -rf /tmp/pyspline

# ---------- Patch transforms3d for numpy >= 1.24 safety ----------
# np.float was removed in NumPy 1.24. With numpy 1.23.5 this is not strictly
# needed, but we patch anyway so the image stays forward-compatible.
RUN TRANSFORMS3D_DIR=$(python3 -c "import transforms3d, os; print(os.path.dirname(transforms3d.__file__))") && \
    find "$TRANSFORMS3D_DIR" -name '*.py' -exec \
        sed -i 's/\bnp\.float\b/float/g' {} +

# ---------- Workspace setup ----------
ENV ROS_WS=/ros2_ws
RUN mkdir -p ${ROS_WS}/src
WORKDIR ${ROS_WS}

# ---------- Auto-source ROS2 in every bash shell ----------
RUN echo 'export LIBGL_ALWAYS_SOFTWARE=1' >> /root/.bashrc && \
    echo 'export MESA_GL_VERSION_OVERRIDE=3.3' >> /root/.bashrc && \
    echo 'source /opt/ros/foxy/setup.bash' >> /root/.bashrc && \
    echo '[ -f ${ROS_WS}/install/setup.bash ] && source ${ROS_WS}/install/setup.bash' >> /root/.bashrc

# ---------- Entrypoint ----------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
