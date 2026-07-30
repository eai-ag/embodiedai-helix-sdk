from embodiedai_helix_sdk import Helix
import time
HOST = "eai-helix-pi-3.local"

def main():
    helix = Helix("100.96.36.10")
    helix.connect()

    helix.command_configuration(
        interface_names=['segment1_dx', 'segment1_dy', 'segment1_l'],
        values=[0.0, 0.0, 0.2]
    )
    time.sleep(5)
    helix.disconnect()


if __name__ == "__main__":
    main()