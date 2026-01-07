

# Additional infos

## Robot Operational States

The Helix robot operates in different system states:

Booting: white: starting systems
blue: initalized- **INITIALIZED**: Robot is ready, but tendon lengths are not calibrated. in order to command tendon lenghts, configuration or cartestian, the robot must restore its calibration pose.

blue blinking: restoring calibration pose. - **RESTORING_CALIBRATION_POSE**: Robot is returning to calibration pose.

running: green - **RUNNING**: Robot is active and will execute commands.

Error: blinking red
Emergency button: solid red

### Error states

Errors occur to prevent overload and protect the system. When the system experiences an error it goes to the error state. 

### Emergency button
pushing the emergency button interrupts the execution of any further commands, the robots remains in its current pose.

when releasing the emergency button, the robot goes back t initialized state and through pushing the button is restores the calibration pose and becomes operational again.


## Connectivity

You can connect to the robot via Ethernet. It is discoverable on a local network by its name: eai-helix-X.local (e.g. eai-helix-0.local). 

Each robot is also configured to have a static IP address; 192.168.238.10X (.e.g 192.168.238.101).

In addition, each robot tries to obatin a dynamic ip address via DHCP lease. You need to ask your network administrator to find it's dyamic IP address. 


## Web interface

The robot hosts a web interface that you can use to inspect it. Open 
http://robot-address or name.local, e.g. http://192.168.xx/ or http://eai-helix-0.local/) in Google Chrome or Chromium to open the web interface. 
In the webinterface you have two tabs (main and debug). main visualizes the current state and let's you publish tendon lenghts, configuration (both through the message panels), and cartesion commands (throught the joysticks at the bottom right)


## Calibration

The robot comes pre-calibrated, i.e. when pressing the button or calling the arm() method, the robot goes to a straight configuraiton. However, it can be necessary to recalibrate this initial pose. This is possible through the web interface. 
Open the web interface, go to the debug tab, and click start calibration.. the motors are now under a little bit of tension. you can manually put the robot into a straight configuration. once in a straight configuration, you can click the finish_calibration button to store this new calibration pose.



## Low Level access

We allow you to manually bypass almost all our safety mechanism. and the SDK provides you the ability to control the motors directly in different modes: current, velocity, position, extended position. 

The emergency button still stops the robot from any further motion (when pressed).

Note, it is risky to control the robot in these modes as it can lead to uncontrolled unwinding of the tendons. 


| Method | Description |
|--------|-------------|
| `set_control_mode(mode: str) -> bool` | Set motor control mode. Valid modes: `"none"`, `"position_control"`, `"velocity_control"`, `"current_control"`, `"extended_position_control"` |


| Method | Description |
|--------|-------------|
| `command_dynamixels(names: List[str], positions=None, velocities=None, efforts=None) -> bool` | Command Dynamixel motors directly. Provide positions, velocities, or efforts based on control mode |


| Method | Returns |
|--------|---------|
| `get_dynamixels_state()` | Dict with `name`, `position`, `velocity`, and `effort` for all 9 dynamixels | -->



```python
# Set control mode for low-level control
helix.set_control_mode("velocity_control")

# Command dynamixels directly
helix.command_dynamixels(
    names=['dynamixel0', 'dynamixel1'],
    velocities=[1.0, -1.0]
)

# Command with position (requires position_control mode)
helix.set_control_mode("position_control")
helix.command_dynamixels(
    names=['dynamixel0'],
    positions=[0.5]
)

# Command with effort/current (requires current_control mode)
helix.set_control_mode("current_control")
helix.command_dynamixels(
    names=['dynamixel0'],
    efforts=[100.0]  # Current in mA
)

# Reset to no control
helix.set_control_mode("none")
```

**Available Control Modes**:
- `"none"`: No motor control
- `"position_control"`: Position control mode
- `"velocity_control"`: Velocity control mode
- `"current_control"`: Current/effort control mode
- `"extended_position_control"`: Extended position control (multi-turn)



```python
dynamixels = helix.get_dynamixels_state()
# Returns: {'name': ['dynamixel0', ...], 'position': [...], 'velocity': [...], 'effort': [...]}

# Access individual dynamixel state
dynamixel_id = 0
position = dynamixels['position'][dynamixel_id]
velocity = dynamixels['velocity'][dynamixel_id]
effort = dynamixels['effort'][dynamixel_id]
``` 










