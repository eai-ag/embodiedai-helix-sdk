# Embodied AI Helix SDK

Python SDK for controlling the Helix continuum robot via ROS bridge.

## Overview

Embodied AI's Helix manipulator is a cable-driven continuum robot with three independently controllable segments. This SDK provides three levels of control abstraction:

- **Tendon Space**: Direct control of the 9 tendons (3 per segment)
- **Configuration Space**: Control each segment's curvature (dx, dy) and length (l)
- **Cartesian Space**: Control the end-effector pose (position + orientation)

## Installation

```bash
pip install embodiedai-helix-sdk
```

Or install from source:

```bash
cd embodiedai-helix-sdk
pip install -e .
```

## Hardware Setup

Follow these steps to prepare the robot:

1. **Power on** - Plug in and turn on the robot
   - Status: Button glows **white** (booting)

2. **Wait for initialization**
   - Status: Button turns **blue** (initialized)

3. **Move to calibration pose** - Press button
   - Status: Button blinks **blue** (restoring calibration pose)

4. **Robot is running**
   - Status: Button turns **green** (ready for commands)

> **Note**: The robot must be in RUNNING state (green button) to accept motion commands.

## Quick Start

```python
from embodiedai_helix_sdk import Helix

# Connect to the robot
helix = Helix("eai-helix-0.local")
helix.connect()

# Command end-effector pose
helix.command_cartesian(
    position=[0.0, 0.0, 0.5],              # x, y, z in meters
    orientation=[0.0, 0.0, 0.0, 1.0]      # quaternion [x, y, z, w]
)

# Read current pose
pose = helix.get_estimated_cartesian()
print(f"Position: {pose['transform']['translation']}")
print(f"Orientation: {pose['transform']['rotation']}")

# Disconnect
helix.disconnect()
```


## Reading Robot State

The robot provides state feedback at three levels of abstraction.

### Tendon Lengths

The lowest-level representation: individual tendon lengths in meters.

**Tendon mapping**:
- Segment 0: tendons 0, 1, 2
- Segment 1: tendons 3, 4, 5
- Segment 2: tendons 6, 7, 8

```python
tendons = helix.get_estimated_tendon_lengths()
# Returns: {'interface_names': ['tendon0', 'tendon1', ..., 'tendon8'],
#           'values': [0.095, 0.098, ...]}

# Access individual tendon
names = tendons['interface_names']
values = tendons['values']
tendon0_length = values[names.index('tendon0')]
```

### Configuration Space

Mid-level representation: each segment's bending and length.

**Parameters per segment**:
- `dx`: Curvature in the x-axis (radians)
- `dy`: Curvature in the y-axis (radians)
- `l`: Segment length (meters)

The configuration is computed from tendon lengths using the constant curvature model.

```python
config = helix.get_estimated_configuration()
# Returns: {'interface_names': ['segment0_dx', 'segment0_dy', 'segment0_l',
#                                'segment1_dx', 'segment1_dy', 'segment1_l',
#                                'segment2_dx', 'segment2_dy', 'segment2_l'],
#           'values': [0.05, 0.02, 0.12, ...]}

# Access specific segment values
names = config['interface_names']
values = config['values']
segment1_dx = values[names.index('segment1_dx')]
segment1_dy = values[names.index('segment1_dy')]
segment1_l = values[names.index('segment1_l')]
```

### Cartesian Pose

Highest-level representation: end-effector position and orientation in 3D space.

The pose is computed via forward kinematics from the configuration of all three segments.

```python
pose = helix.get_estimated_cartesian()
# Returns: TransformStamped message as dict

# Access position (meters)
translation = pose['transform']['translation']
print(f"x: {translation['x']}, y: {translation['y']}, z: {translation['z']}")

# Access orientation (quaternion)
rotation = pose['transform']['rotation']
print(f"qx: {rotation['x']}, qy: {rotation['y']}, qz: {rotation['z']}, qw: {rotation['w']}")
```



## Arming the Robot

When initialized (blue button), the robot can be moved to its calibration pose (blinking blue) and put into RUNNING state (green button) in two ways:
1. **Manually**: Press the button when initialized
2. **Programmatically**: Use the `arm()` method


```python
# Connect to robot
helix = Helix("eai-helix-0.local")
helix.connect()

# Check initial state
if helix.is_initialized():
    print("Robot is initialized but not armed")

# Arm the robot (equivalent to pressing the button)
helix.arm()  # Moves to calibration pose, takes ~10 seconds

# Verify running state
if helix.is_running():
    print("Robot is armed and ready for commands")

    # Execute motion commands
    helix.command_cartesian(
        position=[0.0, 0.0, 0.5],
        orientation=[0.0, 0.0, 0.0, 1.0]
    )

# Disarm when done (equivalent to pressing the button)
helix.disarm()  # Returns to INITIALIZED state

# Disconnect
helix.disconnect()
```



## Commanding Robot Motion

> **Important**: Motion commands are only executed when the robot is in RUNNING state (green button).

### Command Abstraction Levels

The robot accepts commands at three levels of abstraction. 

### Tendon Length Commands

Lowest-level control: directly specify target lengths for individual tendons.

You can command a single tendon, a subset, or all tendons together.

```python
# Control single tendon
helix.command_tendon_lengths(
    interface_names=['tendon6'],
    values=[0.24]
)

# Control one segment (tendons 6, 7, 8)
helix.command_tendon_lengths(
    interface_names=['tendon6', 'tendon7', 'tendon8'],
    values=[0.24, 0.23, 0.19]
)

# Control all tendons
helix.command_tendon_lengths(
    interface_names=['tendon0', 'tendon1', 'tendon2',
                     'tendon3', 'tendon4', 'tendon5',
                     'tendon6', 'tendon7', 'tendon8'],
    values=[0.10, 0.11, 0.12,
            0.20, 0.21, 0.22,
            0.24, 0.23, 0.19]
)
```

**Limit handling**: Commands exceeding physical limits are clamped to valid ranges.
- Segment 0 tendons (0-2): 0.08 - 0.125 m
- Segment 1-2 tendons (3-8): 0.18 - 0.25 m

Violations trigger warnings visible in the robot's Debug logs.

### Configuration Commands

Mid-level control: specify each segment's curvature and length.

Each segment requires all three parameters (dx, dy, l). Segments can be commanded independently or together.

```python
# Control single segment
helix.command_configuration(
    interface_names=['segment1_dx', 'segment1_dy', 'segment1_l'],
    values=[0.05, 0.05, 0.2]
)

# Control all segments
helix.command_configuration(
    interface_names=['segment0_dx', 'segment0_dy', 'segment0_l',
                     'segment1_dx', 'segment1_dy', 'segment1_l',
                     'segment2_dx', 'segment2_dy', 'segment2_l'],
    values=[0.02, 0.01, 0.12,
            0.05, 0.05, 0.20,
            0.03, 0.04, 0.23]
)
```

**Limit handling**: If the commanded configuration produces tendon lengths exceeding limits, all tendons in that segment are scaled proportionally to fit within constraints. Violations trigger warnings.

> **Note**: Partial segment control (e.g., only `segment1_dx`) is not supported. All three parameters must be specified.

### Cartesian Commands

Highest-level control: specify the desired end-effector pose.

The system uses inverse kinematics to compute the required configuration and tendon commands.

```python
# Command target pose
helix.command_cartesian(
    position=[0.0, 0.0, 0.6],           # [x, y, z] in meters
    orientation=[0.0, 0.0, 0.0, 1.0]    # [qx, qy, qz, qw] quaternion
)
```

**Limit handling**: The IK solver computes a configuration as close as possible to the target. The resulting configuration is then scaled if needed, and tendon commands are clamped to physical limits. The robot will achieve the closest reachable pose.





## API Reference

### Connection Management

| Method | Description |
|--------|-------------|
| `Helix(host, port=9090)` | Create connection to Helix robot |
| `connect(timeout=5.0) -> bool` | Establish connection. Returns `True` if successful |
| `disconnect()` | Close connection to robot |
| `is_connected() -> bool` | Check connection status |

### System State Management

| Method | Description |
|--------|-------------|
| `arm()` | Transition to RUNNING state. Moves robot to calibration pose (~10s) |
| `disarm()` | Transition to INITIALIZED state. Stops accepting motion commands |
| `is_running() -> bool` | Check if robot is armed and ready for commands |
| `is_initialized() -> bool` | Check if robot is initialized but not armed |

### State Feedback

All methods return a dictionary with `interface_names` (list of strings) and `values` (list of floats).

| Method | Returns |
|--------|---------|
| `get_estimated_tendon_lengths()` | Current tendon lengths (tendon0-tendon8) in meters |
| `get_estimated_configuration()` | Current configuration (dx, dy, l for each segment) |
| `get_estimated_cartesian()` | Current end-effector pose (TransformStamped with translation and rotation) |

### Motion Commands

> **Requirement**: Robot must be armed (RUNNING state) to execute motion commands.

| Method | Parameters | Description |
|--------|-----------|-------------|
| `command_tendon_lengths(interface_names, values)` | `interface_names`: List[str]<br>`values`: List[float] | Command tendon lengths in meters |
| `command_configuration(interface_names, values)` | `interface_names`: List[str]<br>`values`: List[float] | Command configuration parameters (dx, dy, l) |
| `command_cartesian(position, orientation)` | `position`: [x, y, z]<br>`orientation`: [qx, qy, qz, qw] | Command end-effector pose |



## License

Proprietary - Embodied AI







