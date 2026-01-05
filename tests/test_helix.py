import pytest
from unittest.mock import Mock, patch, MagicMock
from embodiedai_helix_sdk import Helix, ControlMode


class TestHelixInitialization:
    def test_init_with_default_port(self):
        helix = Helix("eai-helix-0.local")
        assert helix.host == "eai-helix-0.local"
        assert helix.port == 9090
        assert helix.client is None

    def test_init_with_custom_port(self):
        helix = Helix("eai-helix-0.local", port=9091)
        assert helix.host == "eai-helix-0.local"
        assert helix.port == 9091

    def test_repr(self):
        helix = Helix("eai-helix-0.local")
        repr_str = repr(helix)
        assert "eai-helix-0.local" in repr_str
        assert "9090" in repr_str
        assert "disconnected" in repr_str


class TestHelixConnection:
    @patch('embodiedai_helix_sdk.helix.roslibpy.Ros')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Service')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Topic')
    def test_connect_success(self, mock_topic, mock_service, mock_ros):
        mock_client = Mock()
        mock_client.is_connected = True
        mock_ros.return_value = mock_client

        helix = Helix("eai-helix-0.local")
        result = helix.connect()

        assert result is True
        assert helix.is_connected() is True
        mock_ros.assert_called_once_with(host="eai-helix-0.local", port=9090)
        mock_client.run.assert_called_once()

    @patch('embodiedai_helix_sdk.helix.roslibpy.Ros')
    def test_connect_failure(self, mock_ros):
        mock_ros.side_effect = Exception("Connection failed")

        helix = Helix("eai-helix-0.local")
        result = helix.connect()

        assert result is False
        assert helix.is_connected() is False

    @patch('embodiedai_helix_sdk.helix.roslibpy.Ros')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Service')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Topic')
    def test_disconnect(self, mock_topic, mock_service, mock_ros):
        mock_client = Mock()
        mock_client.is_connected = True
        mock_ros.return_value = mock_client

        helix = Helix("eai-helix-0.local")
        helix.connect()
        helix.disconnect()

        mock_client.terminate.assert_called_once()
        assert helix.client is None


class TestControlMode:
    @patch('embodiedai_helix_sdk.helix.roslibpy.Ros')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Service')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Topic')
    def test_switch_control_mode_success(self, mock_topic, mock_service_class, mock_ros):
        mock_client = Mock()
        mock_client.is_connected = True
        mock_ros.return_value = mock_client

        mock_service = Mock()
        mock_service.call.return_value = {'success': True}
        mock_service_class.return_value = mock_service

        helix = Helix("eai-helix-0.local")
        helix.connect()

        result = helix.switch_control_mode("position_control")

        assert result is True
        assert helix.get_control_mode() == "position_control"
        mock_service.call.assert_called_once()

    @patch('embodiedai_helix_sdk.helix.roslibpy.Ros')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Service')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Topic')
    def test_switch_control_mode_invalid(self, mock_topic, mock_service_class, mock_ros):
        mock_client = Mock()
        mock_client.is_connected = True
        mock_ros.return_value = mock_client

        helix = Helix("eai-helix-0.local")
        helix.connect()

        with pytest.raises(ValueError, match="Invalid control mode"):
            helix.switch_control_mode("invalid_mode")

    def test_switch_control_mode_not_connected(self):
        helix = Helix("eai-helix-0.local")

        with pytest.raises(ConnectionError, match="Not connected to robot"):
            helix.switch_control_mode("position_control")

    @patch('embodiedai_helix_sdk.helix.roslibpy.Ros')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Service')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Topic')
    def test_all_control_modes(self, mock_topic, mock_service_class, mock_ros):
        mock_client = Mock()
        mock_client.is_connected = True
        mock_ros.return_value = mock_client

        mock_service = Mock()
        mock_service.call.return_value = {'success': True}
        mock_service_class.return_value = mock_service

        helix = Helix("eai-helix-0.local")
        helix.connect()

        valid_modes = ["none", "position_control", "current_control", "velocity_control"]

        for mode in valid_modes:
            result = helix.switch_control_mode(mode)
            assert result is True
            assert helix.get_control_mode() == mode


class TestCommandFunctions:
    @patch('embodiedai_helix_sdk.helix.roslibpy.Ros')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Service')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Topic')
    def test_command_configuration_success(self, mock_topic_class, mock_service_class, mock_ros):
        mock_client = Mock()
        mock_client.is_connected = True
        mock_ros.return_value = mock_client

        mock_service = Mock()
        mock_service.call.return_value = {'success': True}
        mock_service_class.return_value = mock_service

        mock_topic = Mock()
        mock_topic_class.return_value = mock_topic

        helix = Helix("eai-helix-0.local")
        helix.connect()
        helix.switch_control_mode("position_control")

        result = helix.command_configuration(["segment1_dx", "segment1_dy", "segment1_l"], [0.0, 0.0, 0.22])
        assert result is True

    @patch('embodiedai_helix_sdk.helix.roslibpy.Ros')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Service')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Topic')
    def test_command_configuration_wrong_mode(self, mock_topic_class, mock_service_class, mock_ros):
        mock_client = Mock()
        mock_client.is_connected = True
        mock_ros.return_value = mock_client

        mock_service = Mock()
        mock_service.call.return_value = {'success': True}
        mock_service_class.return_value = mock_service

        helix = Helix("eai-helix-0.local")
        helix.connect()
        helix.switch_control_mode("none")

        with pytest.raises(RuntimeError, match="position_control mode"):
            helix.command_configuration(["segment1_dx"], [0.0])

    @patch('embodiedai_helix_sdk.helix.roslibpy.Ros')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Service')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Topic')
    def test_command_tendon_lengths_success(self, mock_topic_class, mock_service_class, mock_ros):
        mock_client = Mock()
        mock_client.is_connected = True
        mock_ros.return_value = mock_client

        mock_service = Mock()
        mock_service.call.return_value = {'success': True}
        mock_service_class.return_value = mock_service

        mock_topic = Mock()
        mock_topic_class.return_value = mock_topic

        helix = Helix("eai-helix-0.local")
        helix.connect()
        helix.switch_control_mode("position_control")

        result = helix.command_tendon_lengths(["tendon6", "tendon7", "tendon8"], [0.24, 0.23, 0.19])
        assert result is True

    @patch('embodiedai_helix_sdk.helix.roslibpy.Ros')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Service')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Topic')
    def test_command_cartesian_success(self, mock_topic_class, mock_service_class, mock_ros):
        mock_client = Mock()
        mock_client.is_connected = True
        mock_ros.return_value = mock_client

        mock_service = Mock()
        mock_service.call.return_value = {'success': True}
        mock_service_class.return_value = mock_service

        mock_topic = Mock()
        mock_topic_class.return_value = mock_topic

        helix = Helix("eai-helix-0.local")
        helix.connect()
        helix.switch_control_mode("position_control")

        result = helix.command_cartesian([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0])
        assert result is True

    @patch('embodiedai_helix_sdk.helix.roslibpy.Ros')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Service')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Topic')
    def test_command_dynamixels_success(self, mock_topic_class, mock_service_class, mock_ros):
        mock_client = Mock()
        mock_client.is_connected = True
        mock_ros.return_value = mock_client

        mock_service = Mock()
        mock_service.call.return_value = {'success': True}
        mock_service_class.return_value = mock_service

        mock_topic = Mock()
        mock_topic_class.return_value = mock_topic

        helix = Helix("eai-helix-0.local")
        helix.connect()
        helix.switch_control_mode("position_control")

        result = helix.command_dynamixels(["dynamixel1", "dynamixel2"], [0.0, 0.0])
        assert result is True


class TestEstimatedStateGetters:
    @patch('embodiedai_helix_sdk.helix.roslibpy.Ros')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Service')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Topic')
    def test_get_estimated_cartesian(self, mock_topic_class, mock_service_class, mock_ros):
        mock_client = Mock()
        mock_client.is_connected = True
        mock_ros.return_value = mock_client

        helix = Helix("eai-helix-0.local")
        helix.connect()

        helix._latest_cartesian = {'transform': {'translation': {'x': 0.1, 'y': 0.2, 'z': 0.3}}}
        cartesian = helix.get_estimated_cartesian()
        assert cartesian == {'transform': {'translation': {'x': 0.1, 'y': 0.2, 'z': 0.3}}}

    @patch('embodiedai_helix_sdk.helix.roslibpy.Ros')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Service')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Topic')
    def test_get_estimated_configuration(self, mock_topic_class, mock_service_class, mock_ros):
        mock_client = Mock()
        mock_client.is_connected = True
        mock_ros.return_value = mock_client

        helix = Helix("eai-helix-0.local")
        helix.connect()

        helix._latest_configuration = {'interface_names': ['segment0_dx', 'segment0_dy', 'segment0_l'], 'values': [0.0, 0.0, 0.22]}
        configuration = helix.get_estimated_configuration()
        assert configuration == {'interface_names': ['segment0_dx', 'segment0_dy', 'segment0_l'], 'values': [0.0, 0.0, 0.22]}

    @patch('embodiedai_helix_sdk.helix.roslibpy.Ros')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Service')
    @patch('embodiedai_helix_sdk.helix.roslibpy.Topic')
    def test_get_estimated_tendon_lengths(self, mock_topic_class, mock_service_class, mock_ros):
        mock_client = Mock()
        mock_client.is_connected = True
        mock_ros.return_value = mock_client

        helix = Helix("eai-helix-0.local")
        helix.connect()

        helix._latest_tendon_lengths = {'interface_names': ['tendon0', 'tendon1', 'tendon2'], 'values': [0.24, 0.23, 0.19]}
        tendon_lengths = helix.get_estimated_tendon_lengths()
        assert tendon_lengths == {'interface_names': ['tendon0', 'tendon1', 'tendon2'], 'values': [0.24, 0.23, 0.19]}


class TestControlModeEnum:
    def test_control_mode_values(self):
        assert ControlMode.NONE.value == "none"
        assert ControlMode.POSITION_CONTROL.value == "position_control"
        assert ControlMode.CURRENT_CONTROL.value == "current_control"
        assert ControlMode.VELOCITY_CONTROL.value == "velocity_control"

    def test_control_mode_from_string(self):
        assert ControlMode("none") == ControlMode.NONE
        assert ControlMode("position_control") == ControlMode.POSITION_CONTROL
        assert ControlMode("current_control") == ControlMode.CURRENT_CONTROL
        assert ControlMode("velocity_control") == ControlMode.VELOCITY_CONTROL
