FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3-pip \
    python3-venv \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-vcstool \
    iproute2 \
    iputils-ping \
    net-tools \
    curl \
    wget \
    nano \
    sudo \
    libyaml-cpp-dev \
    libspdlog-dev \
    libboost-all-dev \
    libeigen3-dev \
    libfmt-dev \
    libglfw3 \
    libglfw3-dev \
    libgl1 \
    libgl1-mesa-dri \
    libglx-mesa0 \
    mesa-utils \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    mujoco==3.3.6 \
    numpy==1.26.4 \
    onnxruntime==1.20.1 \
    PyYAML==6.0.2

ARG UNITREE_RL_LAB_REVISION=4960b84732b0c2ec593dccbfe963fda1bcd7b1e3
RUN git clone --filter=blob:none --no-checkout \
      https://github.com/unitreerobotics/unitree_rl_lab.git /opt/unitree_rl_lab && \
    cd /opt/unitree_rl_lab && \
    git sparse-checkout init --cone && \
    git sparse-checkout set deploy/include deploy/thirdparty \
      deploy/robots/g1_29dof && \
    git checkout "${UNITREE_RL_LAB_REVISION}"

ENV UNITREE_BALANCE_POLICY=/opt/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx
ENV UNITREE_BALANCE_PARAMS=/opt/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml

# Build the same pinned native C++ simulator used by the original project.
# Third-party sources stay in the image and are not committed as company code.
ARG UNITREE_SDK2_REVISION=9754cd153af3da471b0fe5f3aa535e426fb11db3
ARG UNITREE_MUJOCO_REVISION=ae6a8403e272733e9996ef59990880330496177f
ARG MUJOCO_VERSION=3.3.6
RUN git clone https://github.com/unitreerobotics/unitree_sdk2.git /opt/unitree_sdk2 && \
    cd /opt/unitree_sdk2 && git checkout "${UNITREE_SDK2_REVISION}" && \
    cmake -S . -B build -DBUILD_EXAMPLES=OFF \
      -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics && \
    cmake --build build --parallel && cmake --install build && \
    git clone https://github.com/unitreerobotics/unitree_mujoco.git \
      /opt/unitree_mujoco && \
    cd /opt/unitree_mujoco && git checkout "${UNITREE_MUJOCO_REVISION}" && \
    wget -q "https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz" \
      -O /tmp/mujoco.tar.gz && \
    mkdir -p /opt/mujoco && tar -xzf /tmp/mujoco.tar.gz \
      --strip-components=1 -C /opt/mujoco && \
    ln -s /opt/mujoco /opt/unitree_mujoco/simulate/mujoco && \
    sed -i '/#include <mujoco\/mujoco.h>/a #include <algorithm>' \
      /opt/unitree_mujoco/simulate/src/unitree_sdk2_bridge.h && \
    sed -i 's/num_motor_ = mj_model_->nu;/num_motor_ = std::min(mj_model_->nu, 29);/' \
      /opt/unitree_mujoco/simulate/src/unitree_sdk2_bridge.h && \
    sed -i 's/^enable_elastic_band:.*/enable_elastic_band: 1/' \
      /opt/unitree_mujoco/simulate/config.yaml && \
    sed -i 's/^print_scene_information:.*/print_scene_information: 0/' \
      /opt/unitree_mujoco/simulate/config.yaml && \
    cmake -S /opt/unitree_mujoco/simulate \
      -B /opt/unitree_mujoco/simulate/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/unitree_mujoco/simulate/build --parallel && \
    sed -i 's/Instance()->Init(0,/Instance()->Init(1,/' \
      /opt/unitree_rl_lab/deploy/robots/g1_29dof/main.cpp && \
    sed -i 's/^  velocity_commands:/  keyboard_velocity_commands:/' \
      /opt/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml && \
    sed -i \
      -e 's/{"a", {0.0f, 1.0f, 0.0f}}/{"a", {0.0f, 0.80f, 0.0f}}/' \
      -e 's/{"d", {0.0f, -1.0f, 0.0f}}/{"d", {0.0f, -0.80f, 0.0f}}/' \
      /opt/unitree_rl_lab/deploy/robots/g1_29dof/src/State_RLBase.cpp && \
    sed -i '/include_directories(/a\\  /opt/unitree_robotics/include\\n  /opt/unitree_robotics/include/ddscxx' \
      /opt/unitree_rl_lab/deploy/robots/g1_29dof/CMakeLists.txt && \
    sed -i '/link_libraries(/i link_directories(/opt/unitree_robotics/lib)' \
      /opt/unitree_rl_lab/deploy/robots/g1_29dof/CMakeLists.txt && \
    python3 -c "import pathlib,yaml; p=pathlib.Path('/opt/unitree_rl_lab/deploy/robots/g1_29dof/config/config.yaml'); c=yaml.safe_load(p.read_text()); e=c['FSM']['_']; c['FSM']['_']={'Velocity':e['Velocity'],'Passive':e['Passive']}; c['FSM']['Velocity']['transitions']={'Passive':'LT + B.on_pressed'}; c['FSM']['Passive']['transitions']={}; p.write_text(yaml.safe_dump(c,sort_keys=False))" && \
    cmake -S /opt/unitree_rl_lab/deploy/robots/g1_29dof \
      -B /opt/unitree_rl_lab/deploy/robots/g1_29dof/build \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/unitree_rl_lab/deploy/robots/g1_29dof/build \
      --parallel && \
    rm -f /tmp/mujoco.tar.gz

ENV UNITREE_MUJOCO_BIN=/opt/unitree_mujoco/simulate/build/unitree_mujoco
ENV UNITREE_BALANCE_CONTROLLER=/opt/unitree_rl_lab/deploy/robots/g1_29dof/build/g1_ctrl
ENV LD_LIBRARY_PATH=/opt/mujoco/lib:/opt/unitree_robotics/lib:/opt/unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib

# Use one ROS middleware and domain in every development shell. CycloneDDS
# avoids Fast DDS shared-memory transport problems between container shells.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-rmw-cyclonedds-cpp && \
    rm -rf /var/lib/apt/lists/*

ENV ROS_DOMAIN_ID=42
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=file:///workspace/IEA_G1/config/cyclonedds_humble.xml

RUN groupadd --gid ${GROUP_ID} dev && \
    useradd \
        --uid ${USER_ID} \
        --gid ${GROUP_ID} \
        --create-home \
        --shell /bin/bash \
        dev

RUN echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev && \
    chmod 0440 /etc/sudoers.d/dev

RUN echo "source /opt/ros/humble/setup.bash" >> /home/dev/.bashrc

USER dev

WORKDIR /workspace/IEA_G1

CMD ["bash"]
