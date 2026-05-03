#!/bin/bash
set -e

# Start virtual framebuffer + VNC + noVNC so RViz/rqt appear in browser at http://localhost:6080
Xvfb :1 -screen 0 1920x1080x24 &
sleep 1
x11vnc -display :1 -forever -nopw &
sleep 1
websockify --web /usr/share/novnc 6080 localhost:5900 &

export DISPLAY=:1

# Suppress Mesa/libGL warnings (software rendering)
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export GALLIUM_DRIVER=llvmpipe

# Source ROS 2 Foxy
source /opt/ros/foxy/setup.bash

# If the workspace has been built, source the overlay
if [ -f "${ROS_WS}/install/setup.bash" ]; then
    source "${ROS_WS}/install/setup.bash"
fi

exec "$@"
