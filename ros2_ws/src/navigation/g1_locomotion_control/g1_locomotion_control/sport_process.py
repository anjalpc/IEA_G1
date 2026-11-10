"""Supervise the isolated native Sport helper process."""

import math
import select
import subprocess
import sys

from g1_locomotion_control.sport_backend import (
    SportBackend,
    SportBackendError,
    SportProtocolClient,
)


def _wait_readable(stream, timeout):
    return bool(select.select([stream], [], [], timeout)[0])


class SportHelperExchange:
    """Own the native helper subprocess boundary."""

    def __init__(
        self, *, interface, environment,
        response_timeout_seconds,
        rpc_timeout_seconds=None,
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
        if rpc_timeout_seconds is None:
            rpc_timeout_seconds = (
                self.response_timeout_seconds / 2.0
            )
        rpc_timeout_seconds = float(rpc_timeout_seconds)
        if (
            not math.isfinite(rpc_timeout_seconds)
            or rpc_timeout_seconds <= 0.0
        ):
            raise ValueError(
                "rpc_timeout_seconds must be finite and positive"
            )
        self.rpc_timeout_seconds = rpc_timeout_seconds
        if (
            self.response_timeout_seconds
            <= self.rpc_timeout_seconds
        ):
            raise ValueError(
                "response_timeout_seconds must be greater than "
                "rpc_timeout_seconds"
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
                "--rpc-timeout-seconds",
                str(self.rpc_timeout_seconds),
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


class SupervisedSportBackend(SportBackend):
    """Own a Sport backend and its helper transport."""

    def __init__(
        self, client, *, exchange, shutdown_timeout_seconds
    ):
        super().__init__(client)
        self._exchange = exchange
        self._shutdown_timeout_seconds = float(
            shutdown_timeout_seconds
        )

    def shutdown_transport(self):
        """Close and reap the owned helper transport."""
        self._exchange.stop(
            timeout=self._shutdown_timeout_seconds
        )


def create_supervised_sport_backend(
    *, interface, response_timeout_seconds,
    rpc_timeout_seconds,
    environment_factory, exchange_factory=SportHelperExchange,
):
    """Start one helper and return its owned Sport backend."""
    environment = environment_factory()
    exchange = exchange_factory(
        interface=interface,
        environment=environment,
        response_timeout_seconds=response_timeout_seconds,
        rpc_timeout_seconds=rpc_timeout_seconds,
    )
    exchange.start()
    return SupervisedSportBackend(
        SportProtocolClient(exchange),
        exchange=exchange,
        shutdown_timeout_seconds=response_timeout_seconds,
    )
