# Embodied AI Helix SDK

Python SDK for controlling the Helix robot arm via ROS.

## Installation

```bash
pip install embodiedai-helix-sdk
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

```python
from embodiedai_helix_sdk import Helix

helix = Helix("eai-helix-0.local")
helix.connect()

helix.switch_control_mode("position_control")

helix.command_configuration(
    interface_names=['segment0_dx', 'segment0_dy', 'segment0_l'],
    values=[0.0, 0.0, 0.1]
)

config = helix.get_estimated_configuration()
print(config)

helix.disconnect()
```

## Running Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/
```

## API Reference

### Connection

**`Helix(host, port=9090)`**
Create a Helix robot instance.

**`connect(timeout=5.0) -> bool`**
Connect to the robot. Returns True if successful.

**`disconnect()`**
Disconnect from the robot.

**`is_connected() -> bool`**
Check connection status.

### Control Modes

**`switch_control_mode(mode: str) -> bool`**
Switch control mode. Valid modes: `"none"`, `"position_control"`, `"current_control"`, `"velocity_control"`.

**`get_control_mode() -> Optional[str]`**
Get current control mode.

### Command Methods

All command methods require `position_control` mode.

**`command_configuration(interface_names: List[str], values: List[float]) -> bool`**
Command configuration space (dx, dy, L for each segment).

```python
helix.command_configuration(
    interface_names=['segment0_dx', 'segment0_dy', 'segment0_l'],
    values=[0.0, 0.0, 0.1]
)
```

**`command_tendon_lengths(interface_names: List[str], values: List[float]) -> bool`**
Command individual tendon lengths.

```python
helix.command_tendon_lengths(
    interface_names=['tendon0', 'tendon1', 'tendon2'],
    values=[0.1, 0.1, 0.1]
)
```

**`command_cartesian(position: List[float], orientation: List[float]) -> bool`**
Command end-effector pose in Cartesian space.

```python
helix.command_cartesian(
    position=[0.0, 0.0, 0.5],
    orientation=[0.0, 0.0, 0.0, 1.0]
)
```

**`command_dynamixels(interface_names: List[str], values: List[float]) -> bool`**
Command Dynamixel motors directly.

```python
helix.command_dynamixels(
    interface_names=['dynamixel0', 'dynamixel1'],
    values=[0.0, 0.0]
)
```

### State Methods

**`get_estimated_configuration() -> Optional[Dict]`**
Get estimated configuration space state.
Returns: `{'interface_names': [...], 'values': [...]}`

**`get_estimated_tendon_lengths() -> Optional[Dict]`**
Get estimated tendon lengths.
Returns: `{'interface_names': [...], 'values': [...]}`

**`get_estimated_cartesian() -> Optional[Dict]`**
Get estimated Cartesian pose.
Returns: TransformStamped message dict.
