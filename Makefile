# CBF Safety Filter — simulation helpers
# Run from /ros2_ws inside the container.
#
#   make sim          launch simulation (CBF active)
#   make rebuild      clean-build racecar_ece346, then remind you to re-source
#   make reset        put car back at default spawn (2.0, 0.3, yaw=0)
#   make reset X=1 Y=0.5 YAW=1.57   custom position

WS  := /ros2_ws
X   ?= 2.0
Y   ?= 0.3
YAW ?= 0.0

.DEFAULT_GOAL := help
.PHONY: help build rebuild sim sim-off reset value debug binding override on off

help:
	@echo ""
	@echo "  make build        incremental build of racecar_ece346"
	@echo "  make rebuild      clean build from scratch (fixes missing install)"
	@echo "  make sim          launch sim  (CBF intervention ON)"
	@echo "  make sim-off      launch sim  (passthrough, CBF OFF)"
	@echo ""
	@echo "  make reset                   reset car to default (2.0, 0.3, yaw=0)"
	@echo "  make reset X=1 Y=0.5 YAW=0  reset car to custom position"
	@echo ""
	@echo "  make value    echo /safety/value"
	@echo "  make debug    echo /safety/debug_margins  [lane, obs, traf, lookahead]"
	@echo "  make binding  echo /safety/binding_constraint"
	@echo "  make override echo /safety/override"
	@echo ""
	@echo "  make on   enable CBF intervention at runtime"
	@echo "  make off  disable CBF at runtime (passthrough)"
	@echo ""

build:
	cd $(WS) && colcon build --packages-select racecar_ece346
	@echo ""
	@echo "  Done. Run:  source $(WS)/install/setup.bash"

rebuild:
	cd $(WS) && rm -rf build/racecar_ece346 install/racecar_ece346
	cd $(WS) && colcon build --packages-select racecar_ece346
	@echo ""
	@echo "  Done. Run:  source $(WS)/install/setup.bash"

sim:
	cd $(WS) && . install/setup.bash && ros2 launch racecar_ece346 cbf_sim_launch.py

sim-off:
	cd $(WS) && . install/setup.bash && ros2 launch racecar_ece346 cbf_sim_launch.py enable_qp:=false

reset:
	ros2 service call /simulation/reset racecar_interface/srv/Reset "{x: $(X), y: $(Y), yaw: $(YAW)}"

value:
	ros2 topic echo /safety/value

debug:
	ros2 topic echo /safety/debug_margins

binding:
	ros2 topic echo /safety/binding_constraint

override:
	ros2 topic echo /safety/override

on:
	ros2 param set /safety_filter_qp_node enable_qp true

off:
	ros2 param set /safety_filter_qp_node enable_qp false
