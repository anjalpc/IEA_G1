FROM isaac-lab-base:latest

WORKDIR /workspace/isaaclab

RUN ./isaaclab.sh -p -m pip install \
      --no-build-isolation \
      "flatdict==4.0.1" \
 && ./isaaclab.sh -p -m pip install \
      --no-build-isolation \
      --editable \
      /workspace/isaaclab/source/isaaclab \
 && ./isaaclab.sh -p -m pip show flatdict \
 && ./isaaclab.sh -p -m pip show isaaclab \
 && ./isaaclab.sh -p -c \
      "import isaaclab; print('VALID: isaaclab import OK')"
