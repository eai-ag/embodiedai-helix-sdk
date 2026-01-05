# Embodied AI Helix SDK

Python SDK to control Embodied AI's Helix robot arm.

**Package name:** `embodiedai-helix-sdk`
**Import name:** `embodiedai_helix_sdk`

## Installation

```bash
uv pip install -e .
```

## Quick Start

```python
from embodiedai_helix_sdk import Helix

# Connect to robot
helix = Helix("eai-helix-0.local")
helix.connect()

# Switch to position control mode
helix.switch_control_mode("position_control")

# Command the robot
helix.command_configuration(
    interface_names=['segment0_dx', 'segment0_dy', 'segment0_l'],
    values=[0.0, 0.0, 0.1]
)

# Get estimated state
config = helix.get_estimated_configuration()
print(f"Configuration: {config}")

# Disconnect
helix.disconnect()
```

## Control Modes

- `"none"` - No control
- `"position_control"` - Position control mode (required for sending commands)
- `"current_control"` - Current control mode
- `"velocity_control"` - Velocity control mode

## API Reference

### Command Methods

All command methods require the robot to be in `"position_control"` mode.

#### `command_configuration(interface_names, values)`
Command the robot in configuration space (dx, dy, L for each segment).

```python
helix.command_configuration(
    interface_names=['segment0_dx', 'segment0_dy', 'segment0_l',
                     'segment1_dx', 'segment1_dy', 'segment1_l',
                     'segment2_dx', 'segment2_dy', 'segment2_l'],
    values=[0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.2]
)
```

#### `command_tendon_lengths(interface_names, values)`
Command individual tendon lengths.

```python
helix.command_tendon_lengths(
    interface_names=['tendon0', 'tendon1', 'tendon2'],
    values=[0.1, 0.1, 0.1]
)
```

#### `command_cartesian(position, orientation)`
Command the end-effector pose in Cartesian space.

```python
helix.command_cartesian(
    position=[0.0, 0.0, 0.5],  # [x, y, z] in meters
    orientation=[0.0, 0.0, 0.0, 1.0]  # [x, y, z, w] quaternion
)
```

#### `command_dynamixels(interface_names, values)`
Command individual Dynamixel motors directly.

```python
helix.command_dynamixels(
    interface_names=['dynamixel0', 'dynamixel1', 'dynamixel2'],
    values=[0.0, 0.0, 0.0]
)
```

### State Methods

#### `get_estimated_configuration()`
Returns the estimated configuration space state.

```python
config = helix.get_estimated_configuration()
# Returns: {'interface_names': [...], 'values': [...]}
```

#### `get_estimated_tendon_lengths()`
Returns the estimated tendon lengths.

```python
tendons = helix.get_estimated_tendon_lengths()
# Returns: {'interface_names': [...], 'values': [...]}
```

#### `get_estimated_cartesian()`
Returns the estimated Cartesian pose.

```python
pose = helix.get_estimated_cartesian()
# Returns TransformStamped message dict with 'transform' and 'header' fields
```

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Requirements

- Python >= 3.7
- roslibpy >= 1.0.0
- ROSBridge server running on robot (port 9090)
