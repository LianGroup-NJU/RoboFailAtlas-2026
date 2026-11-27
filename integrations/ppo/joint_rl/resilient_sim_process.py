"""Recover a failed scene reset without replaying any policy action.

Only a transport failure or the observed MPLib pybind11 GIL error is retryable.
The replacement process receives exactly the original reset candidate stream.
Other simulator failures, including every failure during step(), remain fatal.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import time
from multiprocessing.connection import Connection


class SimProcess:
    RESET_RETRIES = 2
    RESPONSE_TIMEOUT = 900
    TRANSPORT_ERRORS = (EOFError, BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

    def __init__(self, work: Path, repo: str, python: str, gpu: str):
        self.work = Path(work)
        self.repo, self.python, self.gpu = repo, python, gpu
        self.work.mkdir(parents=True, exist_ok=True)
        self.log = (self.work / "simulator.log").open("ab", buffering=0)
        self.connection = None
        self.process = None
        self._pending = None
        self._reset_retries = 0
        self._closed = False
        try:
            self._start()
        except BaseException:
            try:
                self._stop(graceful=False)
            finally:
                self.log.close()
                self._closed = True
            raise

    def _start(self):
        parent, child = socket.socketpair()
        try:
            self.connection = Connection(parent.detach())
            env = dict(os.environ)
            # Keep the simulator in its original Python environment and GPU.
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            env["PATH"] = str(Path(self.python).parent) + ":" + env.get("PATH", "")
            env["WARP_CACHE_PATH"] = str(self.work / "warp_cache")
            env["CUDA_VISIBLE_DEVICES"] = self.gpu
            env["OMP_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            args = [self.python, "-B", "-u", "-m", "joint_rl.sim_worker", "--fd", str(child.fileno()),
                    "--repo", self.repo, "--work", str(self.work), "--gpu", self.gpu]
            self.process = subprocess.Popen(args, env=env, pass_fds=(child.fileno(),),
                                            stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log)
        finally:
            parent.close()
            child.close()
        ready = self._receive_raw()
        if ready.get("ready") is not True:
            raise RuntimeError("Franka_simulator_invalid_startup_response")

    def _receive_raw(self):
        if not self.connection.poll(self.RESPONSE_TIMEOUT):
            raise TimeoutError("Franka_simulator_timeout; inspect simulator.log")
        result = self.connection.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    @classmethod
    def _recoverable(cls, error):
        if isinstance(error, cls.TRANSPORT_ERRORS):
            return True
        return (isinstance(error, RuntimeError)
                and "pybind11::handle::inc_ref()" in str(error)
                and "PyGILState_Check() failure" in str(error))

    def _event(self, event, **fields):
        row = {"event": event, "time_unix": time.time(),
               "work": str(self.work), "reset_attempt": self._reset_retries,
               "request": self._pending, **fields}
        with (self.work / "reset_recovery.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def _recover_reset(self, error):
        # Never reinterpret a rollout/step failure as a failed policy episode.
        if self._pending is None or self._pending.get("operation") != "reset":
            raise error
        while True:
            if not self._recoverable(error):
                raise error
            if self._reset_retries >= self.RESET_RETRIES:
                self._event("simulator_reset_recovery_exhausted", error=repr(error))
                raise RuntimeError("Franka_reset_recovery_exhausted; inspect reset_recovery.jsonl") from error
            self._reset_retries += 1
            self._event("simulator_reset_recovery_start", error=repr(error),
                        previous_pid=getattr(self.process, "pid", None))
            self._stop(graceful=False)
            try:
                self._start()
                self.connection.send(self._pending)
            except Exception as next_error:
                error = next_error
                # A failed startup can own a process/socket too.
                self._stop(graceful=False)
                continue
            self._event("simulator_reset_recovery_resent", pid=self.process.pid)
            return

    def send(self, value):
        if self._closed:
            raise RuntimeError("Franka_simulator_closed")
        if self._pending is not None:
            raise RuntimeError("Franka_simulator_request_already_pending")
        self._pending = dict(value)
        self._reset_retries = 0
        try:
            self.connection.send(self._pending)
        except Exception as error:
            self._recover_reset(error)

    def receive(self):
        if self._pending is None:
            raise RuntimeError("Franka_simulator_no_request_pending")
        while True:
            try:
                result = self._receive_raw()
            except Exception as error:
                self._recover_reset(error)
                continue
            if self._reset_retries:
                self._event("simulator_reset_recovery_succeeded", pid=self.process.pid)
            self._pending = None
            self._reset_retries = 0
            return result

    def _stop(self, *, graceful):
        process, connection = self.process, self.connection
        try:
            if process is not None and process.poll() is None:
                if graceful and connection is not None:
                    try:
                        connection.send({"operation": "close"})
                        process.wait(timeout=15)
                    except self.TRANSPORT_ERRORS + (subprocess.TimeoutExpired, OSError):
                        pass
                if process.poll() is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=10)
        finally:
            if connection is not None:
                connection.close()
            self.connection = None
            self.process = None

    def close(self):
        if self._closed:
            return
        try:
            self._stop(graceful=True)
        finally:
            self.log.close()
            self._closed = True
