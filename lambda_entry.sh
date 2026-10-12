#!/bin/bash
source /opt/ros/humble/setup.bash
source /ocd/src_repo/install/setup.bash
exec python3 -m awslambdaric "$@"
