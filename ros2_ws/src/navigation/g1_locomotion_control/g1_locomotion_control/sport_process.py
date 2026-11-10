"""Supervise the isolated native Sport helper process."""

import math
import select
import subprocess
import sys

from g1_locomotion_control.sport_backend import SportBackendError


def _wait_readable(stream, timeout):
    return bool(select.select([stream], [], [], timeout)[0])


class SportHelperExchange:
    """Own the native helper subprocess boundary."""

    def __init__(
        self, *, interface, environment,
        response_timeout_seconds,
        process_factory=subprocess.Popen,
        wait_readable=_wait_readable,
    ):
        self.interface = str(interface)
        self.environment = dict(environment)
        response_timeout_seconds = float(response_timeout_seconds)
        if (
            not math.isfinite(response_timeout_seconds)
            or response_timeout_seconds <= 0.0
        ):
            raise ValueError(
                "response_timeout_seconds must be finite and positive"
            )
        self.response_timeout_seconds = float(
            response_timeout_seconds
        )
        self.process_factory = process_factory
        self.wait_readable = wait_readable
        self._process = None

    def start(self):
        """Start exactly one isolated helper process."""
        if (
            self._process is not None
            and self._process.poll() is None
        ):
            raise RuntimeError("Sport helper is already running")
        process = self.process_factory(
            [
                sys.executable,
                "-m",
                "g1_locomotion_control.sport_helper",
                self.interface,
            ],
            env=self.environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None:
            process.terminate()
            try:
                process.wait(
                    timeout=self.response_timeout_seconds
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(
                    timeout=self.response_timeout_seconds
                )
            raise RuntimeError("Sport helper pipes unavailable")
        self._process = process

    def __call__(self, line):
        """Exchange one line or fail closed on timeout."""
        process = self._process
        if process is None or process.poll() is not None:
            raise SportBackendError("SPORT_HELPER_UNAVAILABLE")
        process.stdin.write(str(line) + chr(10))
        process.stdin.flush()
        if not self.wait_readable(
            process.stdout, self.response_timeout_seconds
        ):
            self.stop(timeout=self.response_timeout_seconds)
            raise SportBackendError("SPORT_HELPER_TIMEOUT")
        response = process.stdout.readline()
        if not response:
            self.stop(timeout=self.response_timeout_seconds)
            raise SportBackendError("SPORT_HELPER_UNAVAILABLE")
        return response

    def stop(self, timeout):
        """Terminate and reap the helper process."""
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=timeout)
        else:
            process.wait()
