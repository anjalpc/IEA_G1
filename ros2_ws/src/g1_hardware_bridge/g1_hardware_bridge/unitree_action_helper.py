"""Execute one physical Unitree arm RPC in an isolated DDS process."""

import os
import sys
import traceback


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
    try:
        main()
    except BaseException:  # ensure SDK background threads cannot orphan helper
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
