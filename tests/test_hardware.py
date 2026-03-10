import pytest
import time
from embodiedai_helix_sdk import Helix


@pytest.fixture
def helix(request):
    host = request.config.getoption("--host")
    robot = Helix(host)
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


class TestFTSensor:
    def test_get_ft_sensor_wrench(self, helix):
        time.sleep(0.3)
        wrench = helix.get_ft_sensor_wrench()
        assert wrench is not None
        assert isinstance(wrench, dict)
        assert "force" in wrench
        assert "torque" in wrench
        assert all(axis in wrench["force"] for axis in ["x", "y", "z"])
        assert all(axis in wrench["torque"] for axis in ["x", "y", "z"])
        for axis in ["x", "y", "z"]:
            assert isinstance(wrench["force"][axis], float)
            assert isinstance(wrench["torque"][axis], float)

    def test_get_ft_sensor_temperature(self, helix):
        time.sleep(0.3)
        temperature = helix.get_ft_sensor_temperature()
        assert temperature is not None
        assert isinstance(temperature, float)
        assert -40.0 < temperature < 85.0

    def test_ft_sensor_reset(self, helix):
        time.sleep(0.3)
        result = helix.ft_sensor_reset()
        assert result is True

    def test_ft_sensor_wrench_near_zero_after_reset(self, helix):
        time.sleep(0.3)

        helix.ft_sensor_reset()
        time.sleep(0.5)

        wrench_after = helix.get_ft_sensor_wrench()
        assert wrench_after is not None
        # After reset, force/torque values should be near zero
        for axis in ["x", "y", "z"]:
            assert abs(wrench_after["force"][axis]) < 1.0, f"force.{axis} not near zero after reset"
            assert abs(wrench_after["torque"][axis]) < 1.0, f"torque.{axis} not near zero after reset"


class TestCamera:
    def test_get_image(self, helix):
        image = helix.get_image()
        assert image is not None
        assert image.size[0] > 0
        assert image.size[1] > 0
        assert image.mode == "RGB"

    def test_get_image_returns_consistent_size(self, helix):
        image1 = helix.get_image()
        image2 = helix.get_image()
        assert image1 is not None
        assert image2 is not None
        assert image1.size == image2.size


class TestGripper:
    def test_gripper_open(self, helix):
        result = helix.gripper_open()
        assert result is True

    def test_gripper_close(self, helix):
        result = helix.gripper_close()
        assert result is True

    def test_gripper_set_position(self, helix):
        result = helix.gripper_set_position(0.5)
        assert result is True

    def test_gripper_set_position_fully_open(self, helix):
        result = helix.gripper_set_position(1.0)
        assert result is True

    def test_gripper_set_position_fully_closed(self, helix):
        result = helix.gripper_set_position(0.0)
        assert result is True

    def test_gripper_set_position_invalid_too_high(self, helix):
        with pytest.raises(ValueError, match="position must be between 0.0"):
            helix.gripper_set_position(1.5)

    def test_gripper_set_position_invalid_too_low(self, helix):
        with pytest.raises(ValueError, match="position must be between 0.0"):
            helix.gripper_set_position(-0.1)

    def test_gripper_open_close_cycle(self, helix):
        assert helix.gripper_open() is True
        time.sleep(1.0)
        assert helix.gripper_close() is True
        time.sleep(1.0)
        assert helix.gripper_open() is True


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
