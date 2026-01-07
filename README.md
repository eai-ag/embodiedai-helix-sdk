# Embodied AI Helix SDK

Python SDK for controlling the Helix continuum robot via ROS bridge (roslibpy).

## Installation

```bash
pip install embodiedai-helix-sdk
```

Or install from source:

```bash
cd embodiedai-helix-sdk
pip install -e .
```



## Quick Start

Hardware preparation
Plugin/power/turn on the robot: robot is booting (button is white)  

Wait until it switches from booting to initalized (button is blue): all nodes are started and runnig (though the robot was not put into calibration pose.)  

Press the button to restore the calibration pose (button blinks blue) and transition to running state (button is green).

```python
from embodiedai_helix_sdk import Helix

# Connect to the robot
helix = Helix("eai-helix-0.local")
helix.connect()

# Command the robot in cartesian space
helix.command_cartesian(
    interface_names=['x???', ???'],
    values=[0.0, 0.0, 0.22]
)

# Read cartesian pose
config = helix.XXX()
print(f"Current pose: {config}")

helix.disconnect()
```


## Reading Robot State


### Tendon Lengths

Each segment is controlled by three tendons. segment 0 through tendons with ids 0 1 2. seg 1 with 3 4 5 and seg 2 with 6 7 8

```python
tendons = helix.get_estimated_tendon_lengths()
# Returns: {'interface_names': ['tendon0', 'tendon1', ...], 'values': [...]}

# All 9 tendons: tendon0 through tendon8
```

### Configuration Space

Add description here what the configuraiton space is: dx and dy represent the curvature in the x axis and y axis, l represents the segment length. the configuration is computed based on the tendon lenghts. the robot is composed of three segments with ids 0, 1 and 2. 

```python
config = helix.get_estimated_configuration()
# Returns: {'interface_names': ['segment0_dx', 'segment0_dy', ...], 'values': [...]}

# Access specific values
names = config['interface_names']
values = config['values']
segment1_dx = values[names.index('segment1_dx')]
```

### Cartesian Pose

From the concatenation of the configuration of the robot's three segments 0,1,2 we can compute and obtain the cartesian pose of the robot's tip/end effector. 

```python
pose = helix.get_estimated_cartesian()
# Returns: TransformStamped message as dict

translation = pose['transform']['translation']  # {'x': ..., 'y': ..., 'z': ...}
rotation = pose['transform']['rotation']        # {'x': ..., 'y': ..., 'z': ..., 'w': ...} (quaternion)
```



## Control Robot State

In RUNNIG  state (green button)  the robot can be controlled through tendon length, configuration and cartesian commands.  
**Important**: These high-level commands are only executed when the robot is in RUNNING state (armed).

### Tendon Lenght Commands

It is possible to control individual tendons or a subset of tendons or all tendons together. 

```python
# single tendon
helix.command_tendon_lengths(
    interface_names=['tendon6'],
    values=[0.24]
)

# subset of tendons
helix.command_tendon_lengths(
    interface_names=['tendon6', 'tendon7', 'tendon8'],
    values=[0.24, 0.23, 0.19]
)

# all tendons
helix.command_tendon_lengths(
    interface_names=['tendon6', 'tendon7', 'tendon8', 'tendon6', 'tendon7', 'tendon8','tendon6', 'tendon7', 'tendon8'],
    values=[0.24, 0.23, 0.19]
)
```

Note when commanding values over the limits of tendons, they will be clamped by the robot. The limits of the tendons are XXX, XXX, XXX. You can check if you violate the lengths in the Debug window of the webinterface.


### Configuration Commands

The segments can be commanded independetly or combined, but if any segment is controlled all values must be set.

```python
# Controlling segment 1: Possible
helix.command_configuration(
    interface_names=['segment1_dx', 'segment1_dy', 'segment1_l'],
    values=[0.05, 0.05, 0.2]
)

# Controlling all segments: Possible
helix.command_configuration(
    interface_names=['segment1_dx', 'segment1_dy', 'segment1_l', 'segment1_dx', 'segment1_dy', 'segment1_l', 'segment1_dx', 'segment1_dy', 'segment1_l'],
    values=[0.05, 0.05, 0.2, 0.05, 0.05, 0.2, 0.05, 0.05, 0.2]
)

# Controlling segment partially: NOT possible
helix.command_configuration(
    interface_names=['segment1_dx'],
    values=[0.05]
)
```

note, when commanding configurations that exceed the limit of the tendons, the robot will automatically clamp the commanded configurations.  You can check if you violate the lengths in the Debug window of the webinterface.


### Cartesian Commands
Send a target pose for the end effector

```python
# Command Cartesian pose
helix.command_cartesian(
    position=[0.0, 0.0, 0.6],
    orientation=[0.0, 0.0, 0.0, 1.0]  # Quaternion [x, y, z, w]
)

helix.disarm()
```

When the target pose is not reachable, the robot tries to go as close as possible to the pose by clamping the tendon lenghts.





## API Reference

### Connection

| Method | Description |
|--------|-------------|
| `Helix(host, port=9090)` | Create a Helix robot instance |
| `connect(timeout=5.0) -> bool` | Connect to the robot. Returns True if successful |
| `disconnect()` | Disconnect from the robot |
| `is_connected() -> bool` | Check if connected to robot |

### System State

| Method | Description |
|--------|-------------|
| `arm()` | Arm the robot (INITIALIZED → RUNNING). Takes up to 10 seconds to restore the robot's calibration pose |
| `disarm()` | Disarm the robot (RUNNING → INITIALIZED) |
| `is_running() -> bool` | Check if robot is in RUNNING state |
| `is_initialized() -> bool` | Check if robot is in INITIALIZED state |

### State Reading

| Method | Returns |
|--------|---------|
| `get_estimated_configuration()` | Dict with `interface_names` and `values` for configuration space |
| `get_estimated_tendon_lengths()` | Dict with `interface_names` and `values` for tendon lengths |
| `get_estimated_cartesian()` | Dict with TransformStamped message (translation and rotation) |


### Control Commands

**Note**: These commands require the robot to be armed (RUNNING state).

| Method | Description |
|--------|-------------|
| `command_configuration(interface_names: List[str], values: List[float]) -> bool` | Command configuration space (dx, dy, L for each segment) |
| `command_tendon_lengths(interface_names: List[str], values: List[float]) -> bool` | Command individual tendon lengths (tendon0-tendon8) |
| `command_cartesian(position: List[float], orientation: List[float]) -> bool` | Command end-effector pose. Position: [x, y, z], Orientation: [x, y, z, w] quaternion |





## License

[Add license information]







