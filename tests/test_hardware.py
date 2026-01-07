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


class TestSystemState:
    def test_receive_system_state_message(self, helix):
        time.sleep(0.5)
        assert helix._system_state is not None

    def test_transitions_to_running(self, helix):
        time.sleep(0.5)
        helix.arm()
        time.sleep(7)
        assert helix.is_running() is True
        helix.disarm()
        time.sleep(0.5)
        assert helix.is_running() is False


class TestControlModeSwitching:
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


class TestDynamixelCommands:
    def test_receive_dynamixels_state(self, helix):
        time.sleep(0.5)
        dynamixels_state = helix.get_dynamixels_state()
        assert dynamixels_state is not None
        assert "name" in dynamixels_state
        assert "position" in dynamixels_state
        assert "velocity" in dynamixels_state
        assert len(dynamixels_state["name"]) == 9
        for i in range(9):
            assert f"dynamixel{i}" in dynamixels_state["name"]


    def test_move_each_dynamixel_forth_and_back(self, helix):
        time.sleep(0.3)
        helix.disarm()
        helix.set_control_mode("velocity_control")

        for dynamixel_id in range(9):
            dynamixel_name = f"dynamixel{dynamixel_id}"

            initial_state = helix.get_dynamixels_state()
            assert initial_state is not None
            initial_position = initial_state["position"][dynamixel_id]

            helix.command_dynamixels([dynamixel_name], velocities=[1.0])
            time.sleep(0.3)

            mid_state = helix.get_dynamixels_state()
            assert mid_state is not None
            mid_position = mid_state["position"][dynamixel_id]
            assert mid_position > initial_position, f"{dynamixel_name} did not move forward"

            helix.command_dynamixels([dynamixel_name], velocities=[-1.0])
            time.sleep(0.3)

            final_state = helix.get_dynamixels_state()
            assert final_state is not None
            final_position = final_state["position"][dynamixel_id]
            assert final_position < mid_position, f"{dynamixel_name} did not move backward"

            helix.command_dynamixels([dynamixel_name], velocities=[0.0])
            time.sleep(0.3)

        helix.set_control_mode("none")




class TestEstimatedStates:
    def test_get_estimated_tendon_lengths(self, helix):
        time.sleep(0.3)
        tendon_lengths = helix.get_estimated_tendon_lengths()
        assert tendon_lengths is not None
        assert isinstance(tendon_lengths, dict)
        assert "interface_names" in tendon_lengths
        assert "values" in tendon_lengths
        assert len(tendon_lengths["interface_names"]) == len(tendon_lengths["values"])
        required_tendons = {f"tendon{i}" for i in range(9)}
        assert set(tendon_lengths["interface_names"]) == required_tendons

    def test_get_estimated_configuration(self, helix):
        time.sleep(0.3)
        configuration = helix.get_estimated_configuration()
        assert configuration is not None
        assert isinstance(configuration, dict)
        assert "interface_names" in configuration
        assert "values" in configuration
        assert len(configuration["interface_names"]) == len(configuration["values"])
        required_values = {"segment0_dx", "segment0_dy", "segment0_l", "segment1_dx", "segment1_dy", "segment1_l", "segment2_dx", "segment2_dy", "segment2_l"}
        assert set(configuration["interface_names"]) == required_values

    def test_get_estimated_cartesian(self, helix):
        time.sleep(0.3)
        cartesian = helix.get_estimated_cartesian()
        assert cartesian is not None
        assert isinstance(cartesian, dict)
        assert "translation" in cartesian["transform"]
        assert "rotation" in cartesian["transform"]
        translation = cartesian["transform"]["translation"]
        rotation = cartesian["transform"]["rotation"]
        assert all(axis in translation for axis in ["x", "y", "z"])
        assert all(axis in rotation for axis in ["x", "y", "z", "w"])


class TestTendonLengthCommands:
    def test_send_tendon_length_command(self, helix):
        interface_names = ["tendon6", "tendon7", "tendon8"]
        values = [0.24, 0.23, 0.19]

        result = helix.command_tendon_lengths(interface_names, values)
        assert result is True

    def test_send_tendon_length_command_when_disarmed(self, helix):
        time.sleep(0.3)
        if helix.is_running():
            helix.disarm()
            time.sleep(0.3)

        initial_tendons = helix.get_estimated_tendon_lengths()
        assert initial_tendons is not None

        initial_values = dict(zip(initial_tendons["interface_names"], initial_tendons["values"]))

        interface_names = ["tendon6", "tendon7", "tendon8"]
        commanded_values = [0.24, 0.23, 0.19]

        result = helix.command_tendon_lengths(interface_names, commanded_values)
        assert result is True

        time.sleep(0.5)
        final_tendons = helix.get_estimated_tendon_lengths()
        assert final_tendons is not None

        final_values = dict(zip(final_tendons["interface_names"], final_tendons["values"]))

        tolerance = 0.005
        for name in interface_names:
            assert abs(final_values[name] - initial_values[name]) < tolerance, f"{name} moved when disarmed"

    def test_send_tendon_length_command_when_armed(self, helix):
        time.sleep(0.3)
        if not helix.is_running():
            helix.arm()
            time.sleep(7.0)

        assert helix.is_running() is True

        initial_tendons = helix.get_estimated_tendon_lengths()
        assert initial_tendons is not None

        initial_values = dict(zip(initial_tendons["interface_names"], initial_tendons["values"]))

        interface_names = ["tendon6", "tendon7", "tendon8"]
        commanded_values = [0.24, 0.23, 0.19]

        result = helix.command_tendon_lengths(interface_names, commanded_values)
        assert result is True

        time.sleep(1.0)
        final_tendons = helix.get_estimated_tendon_lengths()
        assert final_tendons is not None

        final_values = dict(zip(final_tendons["interface_names"], final_tendons["values"]))

        # Verify tendons are moving towards commanded values
        # Check that the difference between final and commanded is less than initial and commanded
        for name, cmd_val in zip(interface_names, commanded_values):
            initial_error = abs(initial_values[name] - cmd_val)
            final_error = abs(final_values[name] - cmd_val)
            assert final_error < initial_error, f"{name} not moving towards commanded value"

        helix.disarm()
        time.sleep(0.5)


class TestConfigurationCommands:
    def test_send_configuration_command(self, helix):
        interface_names = ["segment1_dx", "segment1_dy", "segment1_l"]
        values = [0.0, 0.0, 0.22]

        result = helix.command_configuration(interface_names, values)
        assert result is True

    def test_send_configuration_command_when_disarmed(self, helix):
        time.sleep(0.3)
        if helix.is_running():
            helix.disarm()
            time.sleep(5.0)

        initial_config = helix.get_estimated_configuration()
        assert initial_config is not None

        initial_values = dict(zip(initial_config["interface_names"], initial_config["values"]))

        interface_names = ["segment1_dx", "segment1_dy", "segment1_l"]
        commanded_values = [0.01, 0.01, 0.22]

        result = helix.command_configuration(interface_names, commanded_values)
        assert result is True

        time.sleep(0.5)
        final_config = helix.get_estimated_configuration()
        assert final_config is not None

        final_values = dict(zip(final_config["interface_names"], final_config["values"]))

        tolerance = 0.005
        for name in interface_names:
            assert abs(final_values[name] - initial_values[name]) < tolerance, f"{name} moved when disarmed"

    def test_send_configuration_command_when_armed(self, helix):
        time.sleep(0.3)
        if not helix.is_running():
            helix.arm()
            time.sleep(7.0)

        assert helix.is_running() is True

        initial_config = helix.get_estimated_configuration()
        assert initial_config is not None

        initial_values = dict(zip(initial_config["interface_names"], initial_config["values"]))

        interface_names = ["segment1_dx", "segment1_dy", "segment1_l"]
        commanded_values = [0.05, 0.05, 0.2]

        result = helix.command_configuration(interface_names, commanded_values)
        assert result is True

        time.sleep(5.0)
        final_config = helix.get_estimated_configuration()
        assert final_config is not None

        final_values = dict(zip(final_config["interface_names"], final_config["values"]))

        for name, cmd_val in zip(interface_names, commanded_values):
            initial_error = abs(initial_values[name] - cmd_val)
            final_error = abs(final_values[name] - cmd_val)
            assert final_error < initial_error, f"{name} not moving towards commanded value"

        helix.disarm()
        time.sleep(0.5)


class TestCartesianCommands:
    def test_send_cartesian_command(self, helix):
        position = [0.0, 0.0, 0.5]
        orientation = [0.0, 0.0, 0.0, 1.0]

        result = helix.command_cartesian(position, orientation)
        assert result is True

    def test_send_cartesian_command_when_disarmed(self, helix):
        time.sleep(0.3)
        if helix.is_running():
            helix.disarm()
            time.sleep(6.0)

        initial_cartesian = helix.get_estimated_cartesian()
        assert initial_cartesian is not None

        initial_translation = initial_cartesian["transform"]["translation"]
        initial_pos = [initial_translation["x"], initial_translation["y"], initial_translation["z"]]

        position = [0.0, 0.0, 0.6]
        orientation = [0.0, 0.0, 0.0, 1.0]

        result = helix.command_cartesian(position, orientation)
        assert result is True

        time.sleep(0.5)
        final_cartesian = helix.get_estimated_cartesian()
        assert final_cartesian is not None

        final_translation = final_cartesian["transform"]["translation"]
        final_pos = [final_translation["x"], final_translation["y"], final_translation["z"]]

        tolerance = 0.01
        for i in range(3):
            assert abs(final_pos[i] - initial_pos[i]) < tolerance, f"Position axis {i} moved when disarmed"

    def test_send_cartesian_command_when_armed(self, helix):
        time.sleep(0.3)
        if not helix.is_running():
            helix.arm()
            time.sleep(7.0)

        assert helix.is_running() is True

        initial_cartesian = helix.get_estimated_cartesian()
        assert initial_cartesian is not None

        initial_translation = initial_cartesian["transform"]["translation"]
        initial_pos = [initial_translation["x"], initial_translation["y"], initial_translation["z"]]

        commanded_position = [0.1, 0.1, 0.6]
        commanded_orientation = [0.0, 0.0, 0.0, 1.0]

        result = helix.command_cartesian(commanded_position, commanded_orientation)
        assert result is True

        time.sleep(4.0)
        final_cartesian = helix.get_estimated_cartesian()
        assert final_cartesian is not None

        final_translation = final_cartesian["transform"]["translation"]
        final_pos = [final_translation["x"], final_translation["y"], final_translation["z"]]

        for i in range(3):
            initial_error = abs(initial_pos[i] - commanded_position[i])
            final_error = abs(final_pos[i] - commanded_position[i])
            assert final_error < initial_error, f"Position axis {i} not moving towards commanded value"

        helix.disarm()
        time.sleep(0.5)




# TODO:
# pre-commit linting/formatting ?
# ty type checking?
# github linting in branches before release
# release and publish to pypi? or how to install the SDK?
# tests
#    define behavior when going over limits
# Flaky connection to the robot?
# CPU/memory usage on the robot
