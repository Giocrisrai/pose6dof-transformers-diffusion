# =============================================================
# TFM Pose 6-DoF — ROS 2 Humble + MoveIt 2 + CoppeliaSim
# Multi-arch: compatible con ARM64 (Apple Silicon) y AMD64
# =============================================================

FROM ros:humble-desktop

LABEL maintainer="Giocrisrai Godoy <giocrisrai@gmail.com>"
LABEL description="Entorno de simulación para TFM Pose 6-DoF"

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# ─── Dependencias del sistema ────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential cmake git wget curl unzip \
    # Python
    python3-pip python3-colcon-common-extensions \
    python3-rosdep python3-vcstool \
    # MoveIt 2
    ros-humble-moveit \
    ros-humble-moveit-visual-tools \
    ros-humble-moveit-ros-planning-interface \
    # Robot descriptions
    ros-humble-ur-description \
    ros-humble-ur-moveit-config \
    # ROS 2 extras
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-joint-state-publisher-gui \
    ros-humble-rviz2 \
    # Graphics (para CoppeliaSim GUI)
    libgl1-mesa-glx libgl1-mesa-dri \
    mesa-utils xauth \
    # Networking
    net-tools iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# ─── Python packages ─────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    numpy scipy matplotlib opencv-python-headless \
    open3d trimesh pytransform3d \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ─── CoppeliaSim EDU ─────────────────────────────────────────
# Nota: CoppeliaSim no tiene imagen Docker oficial.
# Descargar manualmente y montar como volumen, o usar el instalador.
# Para ARM64 (Apple Silicon), usar la versión compatible.
ENV COPPELIASIM_ROOT=/opt/CoppeliaSim
RUN mkdir -p ${COPPELIASIM_ROOT}

# ─── Workspace ROS 2 ─────────────────────────────────────────
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# ─── Entrypoint ──────────────────────────────────────────────
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
