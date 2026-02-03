# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

Example teleoperation with bimanual Yam:

```shell
# First, start the server processes (in a separate terminal):
# python -m lerobot.scripts.setup_bi_yam_servers

lerobot-teleoperate \
  --robot.type=bi_yam_follower \
  --robot.left_arm_port=1235 \
  --robot.right_arm_port=1234 \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=bi_yam_leader \
  --teleop.left_arm_port=5002 \
  --teleop.right_arm_port=5001 \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    bi_yam_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    bi_yam_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from threading import Lock, Thread
from typing import Any


class BackgroundCameraReader:
    """Reads camera observations in a background thread without blocking the control loop.

    Supports timestamp-based synchronization for data collection by tracking when each
    frame was captured.
    """

    def __init__(self, robot: Robot):
        self.robot = robot
        self.latest_camera_obs: dict[str, Any] = {}
        self.latest_timestamps: dict[str, float] = {}  # Timestamps for each camera
        self.lock = Lock()
        self.stop_event = False
        self.thread: Thread | None = None

    def start(self) -> None:
        """Start the background camera reading thread."""
        self.stop_event = False
        self.thread = Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the background camera reading thread."""
        self.stop_event = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

    def _read_loop(self) -> None:
        """Continuously read camera observations in the background."""
        while not self.stop_event:
            try:
                if hasattr(self.robot, "get_camera_observation_with_timestamps"):
                    camera_obs, timestamps = self.robot.get_camera_observation_with_timestamps()
                    with self.lock:
                        self.latest_camera_obs = camera_obs
                        self.latest_timestamps = timestamps
                elif hasattr(self.robot, "get_camera_observation"):
                    camera_obs = self.robot.get_camera_observation()
                    with self.lock:
                        self.latest_camera_obs = camera_obs
            except Exception as e:
                logging.warning(f"Background camera read error: {e}")
                time.sleep(0.01)  # Small sleep on error to prevent tight loop

    def get_latest(self) -> dict[str, Any]:
        """Get the latest camera observations without blocking."""
        with self.lock:
            return self.latest_camera_obs.copy()

    def get_latest_with_timestamps(self) -> tuple[dict[str, Any], dict[str, float]]:
        """Get the latest camera observations and their timestamps without blocking.

        Returns:
            tuple containing:
                - dict of camera observations (cam_key -> frame)
                - dict of timestamps (cam_key -> timestamp)
        """
        with self.lock:
            return self.latest_camera_obs.copy(), self.latest_timestamps.copy()


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    # Check if robot supports separate camera observation for non-blocking control
    has_separate_camera_obs = hasattr(robot, "get_camera_observation")

    # Start background camera reader if display_data is enabled and robot supports it
    # This ensures camera reading never blocks the control loop
    bg_camera_reader: BackgroundCameraReader | None = None
    if display_data and has_separate_camera_obs:
        bg_camera_reader = BackgroundCameraReader(robot)
        bg_camera_reader.start()

    try:
        while True:
            loop_start = time.perf_counter()

            # Get robot observation (without cameras - they're read in background)
            if has_separate_camera_obs:
                obs = robot.get_observation(include_cameras=False)
            else:
                obs = robot.get_observation()

            # Get teleop action
            raw_action = teleop.get_action()

            # Process teleop action through pipeline
            teleop_action = teleop_action_processor((raw_action, obs))

            # Process action for robot through pipeline
            robot_action_to_send = robot_action_processor((teleop_action, obs))

            # Send processed action to robot (robot_action_processor.to_output should return dict[str, Any])
            _ = robot.send_action(robot_action_to_send)
            action_timestamp = time.perf_counter()  # Record when action was sent

            if display_data:
                # Get latest camera observations from background thread (non-blocking)
                if bg_camera_reader is not None:
                    camera_obs, _ = bg_camera_reader.get_latest_with_timestamps()
                    obs.update(camera_obs)

                # Process robot observation through pipeline
                obs_transition = robot_observation_processor(obs)

                log_rerun_data(
                    observation=obs_transition,
                    action=teleop_action,
                )

                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                # Display the final robot action that was sent
                for motor, value in robot_action_to_send.items():
                    print(f"{motor:<{display_len}} | {value:>7.2f}")
                move_cursor_up(len(robot_action_to_send) + 5)

            dt_s = time.perf_counter() - loop_start
            busy_wait(1 / fps - dt_s)
            loop_s = time.perf_counter() - loop_start
            print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

            if duration is not None and time.perf_counter() - start >= duration:
                return
    finally:
        # Stop background camera reader when loop ends
        if bg_camera_reader is not None:
            bg_camera_reader.stop()


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_devices()
    teleoperate()


if __name__ == "__main__":
    main()
