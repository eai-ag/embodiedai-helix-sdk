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

    def test_arm_robot_transitions_to_running(self, helix):
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



# TODO:
# pre-commit linting/formatting ?
# ty type checking?
# github linting in branches before release
# release and publish to pypi? or how to install the SDK?
# tests
#    discard commands when not in operation mode <- todo 2
#    check if commands converge <- todo 1
#    define behavior when going over limits
# Flaky connection to the robot?
# CPU/memory usage on the robot



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

    # def test_configuration_command_execution(self, helix):
    #     helix.set_control_mode("position_control")
    #     time.sleep(0.5)

    #     interface_names = ["segment1_dx", "segment1_dy", "segment1_l"]
    #     commanded_values = [0.01, 0.02, 0.23]

    #     result = helix.command_configuration(interface_names, commanded_values)
    #     assert result is True

    #     tolerance = 0.01
    #     max_wait_time = 5.0
    #     poll_interval = 0.2
    #     start_time = time.time()

    #     values_match = False
    #     while time.time() - start_time < max_wait_time:
    #         time.sleep(poll_interval)

    #         estimated_config = helix.get_estimated_configuration()
    #         assert estimated_config is not None
    #         assert "interface_names" in estimated_config
    #         assert "values" in estimated_config

    #         estimated_names = estimated_config["interface_names"]
    #         estimated_vals = estimated_config["values"]

    #         all_match = True
    #         for cmd_name, cmd_value in zip(interface_names, commanded_values):
    #             if cmd_name not in estimated_names:
    #                 all_match = False
    #                 break

    #             idx = estimated_names.index(cmd_name)
    #             estimated_value = estimated_vals[idx]

    #             if abs(estimated_value - cmd_value) > tolerance:
    #                 all_match = False
    #                 break

    #         if all_match:
    #             values_match = True
    #             break

    #     assert values_match, f"Configuration values did not converge within {max_wait_time}s"



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

    # def test_cartesian_command_execution(self, helix):
    #     helix.set_control_mode("position_control")
    #     time.sleep(0.5)

    #     commanded_position = [0.15, 0.25, 0.35]
    #     commanded_orientation = [0.0, 0.0, 0.0, 1.0]

    #     result = helix.command_cartesian(commanded_position, commanded_orientation)
    #     assert result is True

    #     tolerance = 0.01
    #     max_wait_time = 5.0
    #     poll_interval = 0.2
    #     start_time = time.time()

    #     values_match = False
    #     while time.time() - start_time < max_wait_time:
    #         time.sleep(poll_interval)

    #         estimated_cartesian = helix.get_estimated_cartesian()
    #         assert estimated_cartesian is not None
    #         assert "transform" in estimated_cartesian
    #         assert "translation" in estimated_cartesian["transform"]
    #         assert "rotation" in estimated_cartesian["transform"]

    #         translation = estimated_cartesian["transform"]["translation"]
    #         rotation = estimated_cartesian["transform"]["rotation"]

    #         assert all(axis in translation for axis in ["x", "y", "z"])
    #         assert all(axis in rotation for axis in ["x", "y", "z", "w"])

    #         position_match = (
    #             abs(translation["x"] - commanded_position[0]) <= tolerance
    #             and abs(translation["y"] - commanded_position[1]) <= tolerance
    #             and abs(translation["z"] - commanded_position[2]) <= tolerance
    #         )

    #         orientation_match = (
    #             abs(rotation["x"] - commanded_orientation[0]) <= tolerance
    #             and abs(rotation["y"] - commanded_orientation[1]) <= tolerance
    #             and abs(rotation["z"] - commanded_orientation[2]) <= tolerance
    #             and abs(rotation["w"] - commanded_orientation[3]) <= tolerance
    #         )

    #         if position_match and orientation_match:
    #             values_match = True
    #             break

    #     assert values_match, f"Cartesian pose did not converge within {max_wait_time}s"
