import time
from embodiedai_helix_sdk import Helix

HOST = "100.96.36.10"


def check(label, ok):
    print(f"[{'OK' if ok else 'FAIL'}] {label}")


def test_connection(helix):
    check("is_connected", helix.is_connected())


def test_system_state(helix):
    time.sleep(0.5)
    check("_system_state received", helix._system_state is not None)

    helix.arm()
    time.sleep(7)
    check("is_running after arm", helix.is_running() is True)
    helix.disarm()
    time.sleep(0.5)
    check("is_running after disarm", helix.is_running() is False)


def test_estimated_states(helix):
    time.sleep(0.3)
    tendon_lengths = helix.get_estimated_tendon_lengths()
    print("get_estimated_tendon_lengths ->", tendon_lengths)

    configuration = helix.get_estimated_configuration()
    print("get_estimated_configuration ->", configuration)

    cartesian = helix.get_estimated_cartesian()
    print("get_estimated_cartesian ->", cartesian)


def test_ft_sensor(helix):
    time.sleep(0.3)
    wrench = helix.get_ft_sensor_wrench()
    print("get_ft_sensor_wrench ->", wrench)

    temperature = helix.get_ft_sensor_temperature()
    print("get_ft_sensor_temperature ->", temperature)

    result = helix.ft_sensor_reset()
    check("ft_sensor_reset", result)

    time.sleep(0.5)
    wrench_after = helix.get_ft_sensor_wrench()
    print("get_ft_sensor_wrench after reset ->", wrench_after)


def test_camera(helix):
    image1 = helix.get_image()
    print("get_image ->", image1.size if image1 else None, image1.mode if image1 else None)

    image2 = helix.get_image()
    check("get_image consistent size", bool(image1) and bool(image2) and image1.size == image2.size)


def test_gripper(helix):
    time.sleep(0.3)
    if not helix.is_running():
        helix.arm()
        time.sleep(7.0)
    check("is_running", helix.is_running())

    check("gripper_open", helix.gripper_open())
    time.sleep(2.0)

    check("gripper_close", helix.gripper_close())
    time.sleep(2.0)

    check("gripper_set_position(0.5)", helix.gripper_set_position(0.5))
    time.sleep(2.0)

    helix.disarm()
    time.sleep(0.5)


def test_tendon_length_commands(helix):
    interface_names = ["tendon6", "tendon7", "tendon8"]
    values = [0.24, 0.23, 0.19]
    check("command_tendon_lengths", helix.command_tendon_lengths(interface_names, values))

    time.sleep(0.3)
    if helix.is_running():
        helix.disarm()
        time.sleep(0.3)
    initial_tendons = helix.get_estimated_tendon_lengths()
    print("tendon lengths before disarmed command ->", initial_tendons)
    helix.command_tendon_lengths(interface_names, values)
    time.sleep(0.5)
    print("tendon lengths after disarmed command ->", helix.get_estimated_tendon_lengths())

    time.sleep(0.3)
    if not helix.is_running():
        helix.arm()
        time.sleep(7.0)
    initial_tendons = helix.get_estimated_tendon_lengths()
    print("tendon lengths before armed command ->", initial_tendons)
    helix.command_tendon_lengths(interface_names, values)
    time.sleep(1.0)
    print("tendon lengths after armed command ->", helix.get_estimated_tendon_lengths())
    helix.disarm()
    time.sleep(0.5)


def test_configuration_commands(helix):
    interface_names = ["segment1_dx", "segment1_dy", "segment1_l"]
    values = [0.0, 0.0, 0.22]
    check("command_configuration", helix.command_configuration(interface_names, values))

    time.sleep(0.3)
    if helix.is_running():
        helix.disarm()
        time.sleep(5.0)
    print("configuration before disarmed command ->", helix.get_estimated_configuration())
    helix.command_configuration(interface_names, [0.01, 0.01, 0.22])
    time.sleep(0.5)
    print("configuration after disarmed command ->", helix.get_estimated_configuration())

    time.sleep(0.3)
    if not helix.is_running():
        helix.arm()
        time.sleep(7.0)
    print("configuration before armed command ->", helix.get_estimated_configuration())
    helix.command_configuration(interface_names, [0.05, 0.05, 0.2])
    time.sleep(5.0)
    print("configuration after armed command ->", helix.get_estimated_configuration())
    helix.disarm()
    time.sleep(0.5)


def test_cartesian_commands(helix):
    position = [0.0, 0.0, 0.5]
    orientation = [0.0, 0.0, 0.0, 1.0]
    check("command_cartesian", helix.command_cartesian(position, orientation))

    time.sleep(0.3)
    if helix.is_running():
        helix.disarm()
        time.sleep(6.0)
    print("cartesian before disarmed command ->", helix.get_estimated_cartesian())
    helix.command_cartesian([0.0, 0.0, 0.6], orientation)
    time.sleep(0.5)
    print("cartesian after disarmed command ->", helix.get_estimated_cartesian())

    time.sleep(0.3)
    if not helix.is_running():
        helix.arm()
        time.sleep(7.0)
    print("cartesian before armed command ->", helix.get_estimated_cartesian())
    helix.command_cartesian([0.1, 0.1, 0.6], orientation)
    time.sleep(4.0)
    print("cartesian after armed command ->", helix.get_estimated_cartesian())
    helix.disarm()
    time.sleep(0.5)


def main():
    helix = Helix(HOST)
    if not helix.connect():
        print(f"Could not connect to robot hardware at {HOST}")
        return

    try:
        test_connection(helix)
        # test_system_state(helix)
        # test_estimated_states(helix)
        test_ft_sensor(helix)
        # test_camera(helix)
        # test_gripper(helix)
        # test_tendon_length_commands(helix)
        # test_configuration_commands(helix)
        # test_cartesian_commands(helix)
    finally:
        helix.disarm()
        helix.disconnect()


if __name__ == "__main__":
    main()
