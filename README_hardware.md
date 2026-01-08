# Helix Hardware Reference

Additional information about the Helix robot hardware, system states, and features.

## Robot Operational States

The Helix robot uses button LED colors to indicate its current state:

| Button Color | State | Description |
|--------------|-------|-------------|
| **White** | BOOTING | System is starting up, ROS nodes are launching |
| **Blue** | INITIALIZED | System ready, but robot is not in calibration pose. Commands will not execute until calibration pose is restored. |
| **Blinking Blue** | RESTORING_CALIBRATION | Robot is moving to calibration pose (~10 seconds) |
| **Green** | RUNNING | Robot is calibrated and will execute motion commands |
| **Blinking Red** | ERROR | System error detected to prevent overload or protect hardware |
| **Solid Red** | EMERGENCY_STOP | Emergency button is pressed, all motion halted |

### State Transitions

**Normal startup sequence:**
1. Power on → BOOTING (white)
2. System ready → INITIALIZED (blue)
3. Press button → RESTORING_CALIBRATION (blinking blue)
4. Calibration complete → RUNNING (green)

**From RUNNING to INITIALIZED:**
- Press button while in RUNNING state
- Call `helix.disarm()` via SDK
- Emergency button pressed and released

### Error States

When an error occurs in the system, the robot enters ERROR state (blinking red).

**Recovery from error:**
- Press the button to acknowledge and clear the error
- Robot returns to INITIALIZED state
- Press button again to restore calibration and return to RUNNING

### Emergency Stop

The emergency button provides immediate motion interruption:

**When pressed (solid red):**
- All motor commands are blocked
- Robot maintains current pose
- System remains powered

**When released:**
- Robot transitions to INITIALIZED state (blue)
- Press button to restore calibration pose and return to RUNNING
- Motion commands can be executed again once RUNNING


## Connectivity

The Helix robot connects via Ethernet and can be accessed through multiple addressing methods.

### Connection Methods

**1. Hostname (Recommended)**
```python
helix = Helix("eai-helix-0.local")
```
- Each robot is discoverable via mDNS/Bonjour using the hostname pattern: `eai-helix-X.local`
- Example: `eai-helix-0.local`, `eai-helix-1.local`, etc.
- Works automatically on most networks without configuration

**2. Static IP Address**
```python
helix = Helix("192.168.238.101")
```
- Each robot has a preconfigured static IP: `192.168.238.10X`
- Example: Robot 1 → `192.168.238.101`, Robot 2 → `192.168.238.102`
- Use this if hostname resolution fails or for direct point-to-point connections

**3. DHCP Dynamic IP**
- Robots also request a dynamic IP via DHCP when connected to a network
- Consult your network administrator or router's DHCP lease table to find the assigned IP
- Use this method for integration into managed networks



## Web Interface

Each Helix robot hosts a built-in web interface for monitoring and manual control.

### Accessing the Web Interface

Open a web browser (Chrome or Chromium recommended) and navigate to:
- `http://eai-helix-0.local` (using hostname)
- `http://192.168.238.101` (using static IP)

Replace with your specific robot's hostname or IP address.

### Interface Tabs

**Main Tab**
- **Status visualization**: Current robot state, button color, and system health
- **State feedback**: Real-time display of tendon lengths, configuration, and cartesian pose
- **Manual control panels**:
  - Tendon length commands (message panel)
  - Configuration commands (message panel)
  - Cartesian commands (joystick widget in bottom-right)

**Debug Tab**
- **System logs**: ROS node outputs and warning messages
- **Limit violations**: Notifications when commands exceed tendon limits
- **Calibration tools**: Start and finish calibration procedures (see Calibration section)
- **Advanced diagnostics**: Motor states, communication status, error details

### Use Cases

- **Monitoring**: View real-time robot state without writing code
- **Manual testing**: Send commands through UI controls to test workspace limits
- **Debugging**: Check logs for limit violations or system errors
- **Calibration**: Perform recalibration procedures when needed


## Calibration

The Helix robot comes pre-calibrated from the factory. The calibration defines the "home pose" - a straight configuration that the robot returns to when armed.

### When to Recalibrate

Recalibration may be necessary if:
- Tendons have been replaced or adjusted
- Robot doesn't return to a straight pose when armed
- Motors have been reconfigured or replaced
- Significant mechanical changes have been made

### Calibration Procedure

Calibration is performed through the web interface:

1. **Open web interface** and navigate to the **Debug tab**

2. **Click "Start Calibration"**
   - Motors switch to current control mode with low constant tension (~50mA)
   - This provides gentle resistance to keep tendons taut without forcing motion
   - Robot becomes manually compliant - you can now move it by hand

3. **Manually straighten the robot**
   - Carefully position all three segments into a straight vertical configuration
   - Ensure tendons have even tension (no slack, but not over-tight)
   - Take your time to achieve accurate alignment

4. **Click "Finish Calibration"**
   - System records current motor positions and encoder values
   - Calibration offsets are saved to persistent storage
   - Motors return to extended position control mode
   - Robot is now ready to arm


### Verifying Calibration

After calibration:
1. Disarm the robot (if armed)
2. Press the button to arm the robot
3. Robot should move to a straight vertical configuration
4. Check that all segments appear aligned and straight

If the robot doesn't achieve a straight pose, repeat the calibration procedure.
