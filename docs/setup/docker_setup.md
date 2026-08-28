# Reproducible Docker development setup

The supported development environment is defined by `docker/Dockerfile` and
Docker Compose. Host installation of ROS 2, MuJoCo, or Python packages is not
required.

## 1. Install host requirements

Use a Linux host with Git, Docker Engine, and Docker Compose v2. Confirm that
the current user can run Docker:

```bash
git --version
docker --version
docker compose version
docker run --rm hello-world
```

Clone the company GitLab project:

```bash
git clone <COMPANY_GITLAB_REPOSITORY_URL> IEA_G1
cd IEA_G1
```

## 2. Start a headless development container

```bash
./scripts/dev.sh
```

The command starts one shared Compose service if necessary and opens a shell.
Run it again from another host terminal to enter the same container. It does
not create another one-off container.

The launcher maps the host UID/GID, builds the image, mounts the checkout at
`/workspace/IEA_G1`, and opens a shell as `dev`. This mode works on hosts
without a display or GPU device.

Verify the environment inside the container:

```bash
whoami
echo "$ROS_DISTRO"
python3 --version
pwd
```

Expected values are `dev`, `humble`, Python `3.10.x`, and
`/workspace/IEA_G1`.

Development shells default to ROS domain `42` with
`rmw_cyclonedds_cpp`, so nodes and service clients started from separate
container terminals use the same DDS configuration. Development DDS is bound
to host loopback through `config/cyclonedds_humble.xml`; it cannot accidentally
discover or command the physical robot network.

## 3. Build and test

Inside the container:

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
colcon test --packages-select g1_robot_interfaces g1_handshake \
  g1_hardware_bridge g1_simulation_bridge \
  --event-handlers console_direct+
colcon test-result --verbose
cd ..

python3 simulation/validate_config.py
python3 simulation/run_g1_model.py --validate-only
python3 simulation/run_g1_model.py --headless --steps 1000
python3 simulation/run_handshake_demo.py --validate-only
python3 simulation/run_handshake_demo.py --headless --duration 0.05
```

A successful `colcon test-result` reports no failed tests. Each simulation
command exits with status zero and prints its validation result.

## 4. Start the interactive viewer

Interactive mode requires Linux X11, a readable Xauthority file, and
`/dev/dri`. From the host repository root:

```bash
./scripts/dev.sh --gui
```

Then, inside the container:

```bash
python3 simulation/run_g1_model.py --native
```

The launcher passes only the current X11 authorization and render devices.
The command prints `glxinfo -B`; stop if the renderer is `llvmpipe`, because
that is software rendering. Hardware renderer names vary by workstation and
are not assumed to be AMD-specific.

## 5. Re-enter or rebuild

Run `./scripts/dev.sh` each time another shell in the shared container is
needed. The checkout is mounted from the host, so source changes persist when
the container exits.

The explicit equivalents are:

```bash
./scripts/dev.sh start  # start without opening a shell
./scripts/dev.sh shell  # open a shell in the shared container
./scripts/dev.sh stop   # stop and remove the shared container
```

After changing the Dockerfile or Compose configuration, exit and rerun the
launcher. To rebuild ROS artifacts without deleting source:

```bash
cd ros2_ws
rm -rf build install log
colcon build --symlink-install
```

All colcon outputs belong under `ros2_ws/`. The `ros2_ws/src` directory is
source and must not be deleted.

## Troubleshooting

- Docker permission errors: configure the host account according to the
  company Docker policy, then start a new login session.
- `--gui` reports no display: use headless mode or start it from an active X11
  desktop session.
- Xauthority is unreadable: set `XAUTHORITY` to the active readable authority
  file before running `./scripts/dev.sh --gui`.

Stop and remove the shared development container after all shells have exited:

```bash
./scripts/dev.sh stop
```
- The viewer reports `llvmpipe`: verify host graphics drivers and Docker access
  to `/dev/dri`; headless validation remains available.
- ROS packages are not found: source `/opt/ros/humble/setup.bash`, then source
  the generated `ros2_ws/install/setup.bash` after building.
