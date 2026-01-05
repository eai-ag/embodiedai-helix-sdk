import roslibpy
from enum import Enum
from typing import Optional, List, Dict, Any

class ControlMode(Enum):
    NONE = "none"
    POSITION_CONTROL = "position_control"
    CURRENT_CONTROL = "current_control"
    VELOCITY_CONTROL = "velocity_control"

class Helix:
    def __init__(self, host: str, port: int = 9090):
        self.host = host
        self.port = port
        self.client: Optional[roslibpy.Ros] = None
        self._set_control_mode_service: Optional[roslibpy.Service] = None
        self._get_control_mode_service: Optional[roslibpy.Service] = None

        self._cmd_cartesian_pub: Optional[roslibpy.Topic] = None
        self._cmd_configuration_pub: Optional[roslibpy.Topic] = None
        self._cmd_tendon_lengths_pub: Optional[roslibpy.Topic] = None

        self._estimated_cartesian_sub: Optional[roslibpy.Topic] = None
        self._estimated_configuration_sub: Optional[roslibpy.Topic] = None
        self._estimated_tendon_lengths_sub: Optional[roslibpy.Topic] = None

        self._latest_cartesian: Optional[Dict] = None
        self._latest_configuration: Optional[Dict] = None
        self._latest_tendon_lengths: Optional[Dict] = None

    def connect(self, timeout: float = 5.0) -> bool:
        try:
            self.client = roslibpy.Ros(host=self.host, port=self.port)
            self.client.run(timeout=timeout)

            self._current_mode: Optional[ControlMode] = None
            self._set_control_mode_service = roslibpy.Service(self.client,'/helix/dynamixel_driver_node/set_control_mode','helix_interfaces/SetString')

            self._cmd_cartesian_pub = roslibpy.Topic(self.client,'/helix/command/cartesian','geometry_msgs/Pose')
            self._cmd_configuration_pub = roslibpy.Topic(self.client,'/helix/command/configuration','control_msgs/InterfaceValue')
            self._cmd_tendon_lengths_pub = roslibpy.Topic(self.client,'/helix/command/tendon_lengths','control_msgs/InterfaceValue')

            self._estimated_cartesian_sub = roslibpy.Topic(self.client,'/helx/estimated/cartesian','geometry_msgs/TransformStamped')
            self._estimated_cartesian_sub.subscribe(self._cartesian_callback)

            self._estimated_configuration_sub = roslibpy.Topic(self.client,'/helix/etimated/configuration','control_msgs/InterfaceValue')
            self._estimated_configuration_sub.subscribe(self._configuration_callback)

            self._estimated_tendon_lengths_sub = roslibpy.Topic(self.client,'/helix/estimated/tendon_lengths','control_msgs/InterfaceValue')
            self._estimated_tendon_lengths_sub.subscribe(self._tendon_lengths_callback)

            return self.is_connected()
        except Exception as e:
            print(f"Failed to connect to {self.host}:{self.port}: {e}")
            return False

    def disconnect(self):
        if self.client and self.client.is_connected:
            if self._estimated_cartesian_sub:
                self._estimated_cartesian_sub.unsubscribe()
            if self._estimated_configuration_sub:
                self._estimated_configuration_sub.unsubscribe()
            if self._estimated_tendon_lengths_sub:
                self._estimated_tendon_lengths_sub.unsubscribe()

            self.client.close()
            self.client = None

            self._control_mode_service = None

            self._cmd_cartesian_pub = None
            self._cmd_configuration_pub = None
            self._cmd_tendon_lengths_pub = None

            self._estimated_cartesian_sub = None
            self._estimated_configuration_sub = None
            self._estimated_tendon_lengths_sub = None

    def is_connected(self) -> bool:
        return self.client is not None and self.client.is_connected


    def set_control_mode(self, mode: str) -> bool:
        if not self.is_connected():
            raise ConnectionError("Not connected to robot. Call connect() first.")

        try:
            control_mode = ControlMode(mode)
        except ValueError:
            valid_modes = [m.value for m in ControlMode]
            raise ValueError(f"Invalid control mode '{mode}'. Valid modes: {valid_modes}")

        try:
            request = roslibpy.ServiceRequest({'data': mode})
            response = self._control_mode_service.call(request, timeout=5.0)

            if response.get('success', False):
                self._current_mode = control_mode
                return True
            else:
                print(f"Failed to switch mode: {response.get('message', 'Unknown error')}")
                return False

        except Exception as e:
            print(f"Error switching control mode: {e}")
            return False

    def _check_position_control(self):
        if self._current_mode != ControlMode.POSITION_CONTROL:
            raise RuntimeError("Commands can only be sent in position_control mode")


    def command_configuration(self, interface_names: List[str], values: List[float]) -> bool:
        if not self.is_connected():
            raise ConnectionError("Not connected to robot. Call connect() first.")

        self._check_position_control()

        if len(interface_names) != len(values):
            raise ValueError("interface_names and values must have the same length")

        try:
            message = {
                'interface_names': interface_names,
                'values': values
            }
            self._cmd_configuration_pub.publish(roslibpy.Message(message))
            return True
        except Exception as e:
            print(f"Error sending configuration command: {e}")
            return False

    def command_tendon_lengths(self, interface_names: List[str], values: List[float]) -> bool:
        if not self.is_connected():
            raise ConnectionError("Not connected to robot. Call connect() first.")

        self._check_position_control()

        if len(interface_names) != len(values):
            raise ValueError("interface_names and values must have the same length")

        try:
            message = {
                'interface_names': interface_names,
                'values': values
            }
            self._cmd_tendon_lengths_pub.publish(roslibpy.Message(message))
            return True
        except Exception as e:
            print(f"Error sending tendon lengths command: {e}")
            return False

    def command_cartesian(self, position: List[float], orientation: List[float]) -> bool:
        if not self.is_connected():
            raise ConnectionError("Not connected to robot. Call connect() first.")

        self._check_position_control()

        if len(position) != 3:
            raise ValueError("position must have 3 values [x, y, z]")
        if len(orientation) != 4:
            raise ValueError("orientation must have 4 values [x, y, z, w] (quaternion)")

        try:
            message = {
                'position': {
                    'x': position[0],
                    'y': position[1],
                    'z': position[2]
                },
                'orientation': {
                    'x': orientation[0],
                    'y': orientation[1],
                    'z': orientation[2],
                    'w': orientation[3]
                }
            }
            self._cmd_cartesian_pub.publish(roslibpy.Message(message))
            return True
        except Exception as e:
            print(f"Error sending cartesian command: {e}")
            return False


    def _cartesian_callback(self, message):
        self._latest_cartesian = message

    def _configuration_callback(self, message):
        self._latest_configuration = message

    def _tendon_lengths_callback(self, message):
        self._latest_tendon_lengths = message

    def get_estimated_cartesian(self) -> Optional[Dict[str, Any]]:
        if self._latest_cartesian:
            return self._latest_cartesian
        return None

    def get_estimated_configuration(self) -> Optional[Dict[str, List[float]]]:
        if self._latest_configuration:
            return {
                'interface_names': self._latest_configuration.get('interface_names', []),
                'values': self._latest_configuration.get('values', [])
            }
        return None

    def get_estimated_tendon_lengths(self) -> Optional[Dict[str, List[float]]]:
        if self._latest_tendon_lengths:
            return {
                'interface_names': self._latest_tendon_lengths.get('interface_names', []),
                'values': self._latest_tendon_lengths.get('values', [])
            }
        return None


    def __repr__(self) -> str:
        status = "connected" if self.is_connected() else "disconnected"
        return f"Helix(host='{self.host}', port={self.port}, status='{status}')"
