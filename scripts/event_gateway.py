#!/usr/bin/env python3
"""Persistent localhost gateway for dashboard handshake/card ROS services."""

import json
import os
import socketserver
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


SOCKET_PATH = os.environ.get(
    "G1_EVENT_SOCKET", "/tmp/g1_event_control.sock"
)
SERVICES = {
    "handshake:start": "/event/handshake/start",
    "handshake:grip": "/event/handshake/grip",
    "handshake:cancel": "/event/handshake/cancel",
    "card:hold": "/event/card/hold",
    "card:present": "/event/card/present",
    "card:cancel": "/event/card/cancel",
}


class EventGateway(Node):
    def __init__(self):
        super().__init__("g1_event_gateway")
        self._event_clients = {
            key: self.create_client(Trigger, service)
            for key, service in SERVICES.items()
        }
        self.command_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.card_state = None
        self.create_subscription(
            String, "/event/card/state", self._card_state_callback, 10
        )

    def _card_state_callback(self, message):
        try:
            state = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if isinstance(state, dict):
            with self.state_lock:
                self.card_state = state

    def request(self, payload):
        if payload.get("type") == "card_state":
            with self.state_lock:
                state = dict(self.card_state) if self.card_state else None
            return {"ok": state is not None, "state": state}

        key = f"{payload.get('group', '')}:{payload.get('action', '')}"
        client = self._event_clients.get(key)
        if client is None:
            return {"ok": False, "error": "unknown event command"}

        with self.command_lock:
            if not client.wait_for_service(timeout_sec=1.0):
                return {"ok": False, "error": "ROS event service unavailable"}
            future = client.call_async(Trigger.Request())
            done = threading.Event()
            future.add_done_callback(lambda _future: done.set())
            if not done.wait(5.0):
                return {"ok": False, "error": "ROS event service timed out"}
            try:
                response = future.result()
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            return {
                "ok": True,
                "success": bool(response.success),
                "message": str(response.message),
            }


class GatewayHandler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            payload = json.loads(self.rfile.readline(8192).decode("utf-8"))
            response = self.server.gateway.request(payload)
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))


class GatewayServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def main():
    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass
    rclpy.init()
    gateway = EventGateway()
    server = GatewayServer(SOCKET_PATH, GatewayHandler)
    server.gateway = gateway
    os.chmod(SOCKET_PATH, 0o600)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    gateway.get_logger().info(f"event gateway ready on {SOCKET_PATH}")
    try:
        rclpy.spin(gateway)
    finally:
        server.shutdown()
        server.server_close()
        gateway.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        try:
            os.unlink(SOCKET_PATH)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
