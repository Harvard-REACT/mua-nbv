#!/usr/bin/env bash
# setup_env.sh — Create and configure the Python virtual environment for MUA-NBV.
#
# Usage:
#   source setup_env.sh          # create venv + install deps + activate
#   source setup_env.sh --skip-install  # activate existing venv only
#
# Prerequisites:
#   - ROS 2 Jazzy installed: source /opt/ros/jazzy/setup.bash
#   - Python 3.10+
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# --system-site-packages lets the venv inherit ROS 2 Python packages
# (rclpy, geometry_msgs, tf2_ros, etc.) that are installed system-wide via apt.
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR} ..."
    python3 -m venv --system-site-packages "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [[ "${1:-}" != "--skip-install" ]]; then
    echo "Installing pip dependencies ..."
    pip install --upgrade pip
    pip install -r "${SCRIPT_DIR}/requirements.txt"
    echo ""
    echo "NOTE: requirements.txt includes numexpr, bottleneck, and scikit-learn"
    echo "      to override system packages compiled against numpy 1.x."
fi

echo ""
echo "Virtual environment active: ${VENV_DIR}"
echo "Python: $(which python3)"
echo ""
echo "Next steps:"
echo "  cd ${SCRIPT_DIR}/ws"
echo "  source /opt/ros/jazzy/setup.bash"
echo "  colcon build --symlink-install"
echo "  source install/setup.bash"
