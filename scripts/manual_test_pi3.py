"""Manual smoke test for the Helix SDK against eai-helix-pi-3.

pi-3 has no gripper (HAS_GRIPPER=false), so gripper functions are skipped.
Run with the robot powered on and armed/arming via the button, or let this
script arm it. Watch the button LED and the robot's motion as each step runs.
"""

import time

from embodiedai_helix_sdk import Helix

HOST = "eai-helix-pi-3.local"


def section(title: str):
    print(f"\n=== {title} ===")


def main():
    helix = Helix(HOST)

    section("Connect")
    if not helix.connect():
        print(f"Failed to connect to {HOST}")
        return
    print(f"Connected: {helix}")

    time.sleep(0.5)
    print(f"System state: {helix._system_state}")

    section("Arm")
    if helix.is_initialized():
        helix.arm()
        time.sleep(7.0)
    print(f"Running: {helix.is_running()}")

    section("Estimated state")
    print("Tendon lengths:", helix.get_estimated_tendon_lengths())
    print("Configuration:", helix.get_estimated_configuration())
    print("Cartesian:", helix.get_estimated_cartesian())

    section("Force-torque sensor")
    print("Wrench:", helix.get_ft_sensor_wrench())
    print("Temperature:", helix.get_ft_sensor_temperature())
    print("Reset:", helix.ft_sensor_reset())

    section("Camera")
    image = helix.get_image()
    if image is not None:
        print(f"Got image: {image.size} {image.mode}")
        image.save("pi3_capture.jpg")
    else:
        print("No image received")

    section("Tendon length command")
    print(helix.command_tendon_lengths(["tendon6", "tendon7", "tendon8"], [0.24, 0.23, 0.19]))
    time.sleep(2.0)

    section("Configuration command")
    print(helix.command_configuration(["segment1_dx", "segment1_dy", "segment1_l"], [0.05, 0.05, 0.2]))
    time.sleep(4.0)

    section("Cartesian command")
    print(helix.command_cartesian(position=[0.0, 0.0, 0.5], orientation=[0.0, 0.0, 0.0, 1.0]))
    time.sleep(4.0)

    section("Disarm")
    helix.disarm()
    time.sleep(0.5)

    section("Disconnect")
    helix.disconnect()
    print(f"Connected: {helix.is_connected()}")


if __name__ == "__main__":
    main()
