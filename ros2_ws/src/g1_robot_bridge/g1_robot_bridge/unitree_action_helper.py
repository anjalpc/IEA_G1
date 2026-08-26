"""Execute one Unitree arm RPC in an isolated DDS domain-0 process."""

import sys


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: helper INTERFACE TIMEOUT_S ACTION_ID")
    interface = sys.argv[1]
    timeout = float(sys.argv[2])
    action_id = int(sys.argv[3])

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient

    ChannelFactoryInitialize(0, interface)
    client = G1ArmActionClient()
    client.SetTimeout(timeout)
    client.Init()
    code = int(client.ExecuteAction(action_id))
    print(f"ACTION_RESULT={code}", flush=True)


if __name__ == "__main__":
    main()
