import pytest
import time
from embodiedai_helix_sdk import Helix


@pytest.fixture
def helix():
    robot = Helix("eai-helix-0.local")
    connected = robot.connect()
    if not connected:
        pytest.skip("Could not connect to robot hardware")
    yield robot
    robot.disconnect()


class TestHardwareConnection:
    def test_connect_to_robot(self, helix):
        assert helix.is_connected()

    def test_switch_to_none_mode(self, helix):
        result = helix.set_control_mode("none")
        assert result is True

    def test_switch_to_current_mode(self, helix):
        result = helix.set_control_mode("current_control")
        assert result is True

    def test_switch_to_velocity_control(self, helix):
        result = helix.set_control_mode("velocity_control")
        assert result is True

    def test_switch_to_position_control(self, helix):
        result = helix.set_control_mode("position_control")
        assert result is True

    def test_invalid_control_mode(self, helix):
        with pytest.raises(RuntimeError, match="Failed to switch to invalid_mode mode"):
            helix.set_control_mode("invalid_mode")






# class TestConfigurationCommands:
#     def test_send_configuration_command(self, helix):
#         helix.set_control_mode("position_control")
#         time.sleep(0.5)

#         interface_names = ["segment1_dx", "segment1_dy", "segment1_l"]
#         values = [0.0, 0.0, 0.22]

#         result = helix.command_configuration(interface_names, values)
#         assert result is True

#     def test_configuration_requires_position_control(self, helix):
#         helix.set_control_mode("none")
#         time.sleep(0.5)

#         interface_names = ["segment1_dx", "segment1_dy", "segment1_l"]
#         values = [0.0, 0.0, 0.22]

#         with pytest.raises(RuntimeError, match="position_control mode"):
#             helix.command_configuration(interface_names, values)


# class TestTendonLengthCommands:
#     def test_send_tendon_length_command(self, helix):
#         helix.set_control_mode("position_control")
#         time.sleep(0.5)

#         interface_names = ["tendon6", "tendon7", "tendon8"]
#         values = [0.24, 0.23, 0.19]

#         result = helix.command_tendon_lengths(interface_names, values)
#         assert result is True

#     def test_tendon_length_requires_position_control(self, helix):
#         helix.set_control_mode("none")
#         time.sleep(0.5)

#         interface_names = ["tendon6", "tendon7", "tendon8"]
#         values = [0.24, 0.23, 0.19]

#         with pytest.raises(RuntimeError, match="position_control mode"):
#             helix.command_tendon_lengths(interface_names, values)


# class TestCartesianCommands:
#     def test_send_cartesian_command(self, helix):
#         helix.set_control_mode("position_control")
#         time.sleep(0.5)

#         position = [0.1, 0.2, 0.3]
#         orientation = [0.0, 0.0, 0.0, 1.0]

#         result = helix.command_cartesian(position, orientation)
#         assert result is True

#     def test_cartesian_requires_position_control(self, helix):
#         helix.set_control_mode("none")
#         time.sleep(0.5)

#         position = [0.1, 0.2, 0.3]
#         orientation = [0.0, 0.0, 0.0, 1.0]

#         with pytest.raises(RuntimeError, match="position_control mode"):
#             helix.command_cartesian(position, orientation)


# class TestDynamixelCommands:
#     def test_send_dynamixel_command(self, helix):
#         helix.set_control_mode("position_control")
#         time.sleep(0.5)

#         interface_names = ["dynamixel1", "dynamixel2"]
#         values = [0.0, 0.0]

#         result = helix.command_dynamixels(interface_names, values)
#         assert result is True

#     def test_dynamixel_requires_position_control(self, helix):
#         helix.set_control_mode("none")
#         time.sleep(0.5)

#         interface_names = ["dynamixel1", "dynamixel2"]
#         values = [0.0, 0.0]

#         with pytest.raises(RuntimeError, match="position_control mode"):
#             helix.command_dynamixels(interface_names, values)


# class TestEstimatedStates:
#     def test_get_estimated_cartesian(self, helix):
#         time.sleep(1.0)
#         cartesian = helix.get_estimated_cartesian()
#         assert cartesian is not None
#         assert isinstance(cartesian, dict)

#     def test_get_estimated_configuration(self, helix):
#         time.sleep(1.0)
#         configuration = helix.get_estimated_configuration()
#         assert configuration is not None
#         assert isinstance(configuration, dict)
#         assert 'interface_names' in configuration
#         assert 'values' in configuration

#     def test_get_estimated_tendon_lengths(self, helix):
#         time.sleep(1.0)
#         tendon_lengths = helix.get_estimated_tendon_lengths()
#         assert tendon_lengths is not None
#         assert isinstance(tendon_lengths, dict)
#         assert 'interface_names' in tendon_lengths
#         assert 'values' in tendon_lengths


# class TestFullWorkflow:
#     def test_complete_workflow(self, helix):
#         helix.set_control_mode("position_control")
#         time.sleep(0.5)

#         config_interface_names = ["segment1_dx", "segment1_dy", "segment1_l"]
#         config_values = [0.0, 0.0, 0.22]
#         helix.command_configuration(config_interface_names, config_values)

#         time.sleep(1.0)

#         tendon_interface_names = ["tendon6", "tendon7", "tendon8"]
#         tendon_values = [0.24, 0.23, 0.19]
#         helix.command_tendon_lengths(tendon_interface_names, tendon_values)

#         time.sleep(1.0)

#         cartesian = helix.get_estimated_cartesian()
#         configuration = helix.get_estimated_configuration()
#         tendon_lengths = helix.get_estimated_tendon_lengths()

#         assert cartesian is not None
#         assert configuration is not None
#         assert tendon_lengths is not None

#         helix.set_control_mode("none")
#         assert helix.get_control_mode() == "none"
