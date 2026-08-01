"""SSH-based remote execution target (SPEC.md §5, §15.3). Reference
implementation for `execution_target.mode: ssh`, validated for real against
a remote host - see VALIDATION.md.

Deliberately does NOT use docker-py's `ssh://` transport: that requires
`paramiko` just to import the module (even in its "shell out to the system
ssh client" mode), and offers no clean way to pin a specific identity file
short of editing `~/.ssh/config` or relying on ssh-agent state this process
has no guarantee persists between invocations. Instead, this shells out to
`ssh -i <key> ...` directly for every docker operation - more code than the
local adapter, but fully under our control and using the exact identity
file a suite config names, no agent/config-file side effects required.

LIMITATION, scoped deliberately for v1: only `source.mode: prebuilt` is
supported. Building from source on a remote host would need the build
context transferred there first (rsync, or accepting docker-py's local-tar
upload path some other way); out of scope for this pass - a suite wanting
`source.mode: build` against a remote host should build locally and push
the resulting image to a registry the remote host can pull from instead,
or this adapter should grow that support later.
"""
from __future__ import annotations

import shlex
import subprocess
from typing import Any

from llapdance.plugins.base import ExecutionTargetAdapter, RunningBackend
from llapdance.plugins.registry import register


class SSHCommandError(RuntimeError):
    pass


class SSHRunningBackend(RunningBackend):
    def __init__(self, container_id: str, endpoint: str, adapter: "SSHDockerExecutionTarget") -> None:
        self._container_id = container_id
        self._endpoint = endpoint
        self._adapter = adapter

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def container_id(self) -> str:
        return self._container_id

    def logs(self, tail: int = 200) -> str:
        return self._adapter._docker(["logs", "--tail", str(tail), self._container_id])


class SSHDockerExecutionTarget(ExecutionTargetAdapter):
    name = "ssh-docker"

    def __init__(self, config: dict[str, Any]) -> None:
        self._host = config["host"]
        self._user = config.get("user")
        self._ssh_key_path = config.get("ssh_key_path")
        target = f"{self._user}@{self._host}" if self._user else self._host

        def ssh_cmd(remote_args: list[str]) -> list[str]:
            cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
            if self._ssh_key_path:
                cmd += ["-i", self._ssh_key_path]
            return cmd + [target, shlex.join(remote_args)]

        self._ssh_cmd = ssh_cmd

    def _docker(self, args: list[str]) -> str:
        proc = subprocess.run(self._ssh_cmd(["docker"] + args), capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            raise SSHCommandError(f"remote `docker {' '.join(args)}` failed: {proc.stderr.strip()}")
        return proc.stdout

    def build(self, backend_config: dict[str, Any]) -> str:
        source = backend_config["source"]
        if source["mode"] != "prebuilt":
            raise NotImplementedError(
                "ssh-docker only supports source.mode: prebuilt (see module docstring) - "
                "build from source locally and push to a registry the remote host can pull from instead"
            )
        image = source["image"]
        try:
            self._docker(["image", "inspect", image])
        except SSHCommandError:
            self._docker(["pull", image])
        return image

    def start(self, backend_config: dict[str, Any], image_ref: str, device_indices: list[int]) -> RunningBackend:
        network_cfg = backend_config.get("network", {})
        network_mode = network_cfg.get("mode", "disabled")
        port = backend_config.get("port", 8000)

        args = ["run", "-d"]
        for key, value in backend_config.get("env", {}).items():
            args += ["-e", f"{key}={value}"]
        for host_path, container_path in backend_config.get("volumes", {}).items():
            args += ["-v", f"{host_path}:{container_path}:ro"]
        for device in backend_config.get("devices", []):
            args += ["--device", device]
        if network_mode == "isolated":
            args += ["--network", "none"]
        elif network_mode == "enabled":
            args += ["--network", network_cfg["network"]]
        if network_mode != "isolated":
            args += ["-p", f"0:{port}"]
        args.append(image_ref)
        args += backend_config.get("command", [])

        container_id = self._docker(args).strip()

        if network_mode == "isolated":
            ip = self._docker(
                ["inspect", "--format", "{{.NetworkSettings.IPAddress}}", container_id]
            ).strip()
            endpoint = f"http://{ip}:{port}"
        else:
            port_out = self._docker(["port", container_id, str(port)]).strip()
            # "0.0.0.0:34567" (one line, IPv4) - take the port after the last ':'
            host_port = port_out.splitlines()[0].rsplit(":", 1)[-1]
            endpoint = f"http://{self._host}:{host_port}"

        return SSHRunningBackend(container_id, endpoint, self)

    def stop(self, backend: RunningBackend) -> None:
        assert isinstance(backend, SSHRunningBackend)
        self._docker(["stop", "--time", "10", backend.container_id])
        self._docker(["rm", "-f", backend.container_id])

    def list_images(self, name_filter: str | None = None) -> list[dict[str, Any]]:
        out = self._docker(["images", "--format", "{{.ID}}\t{{.Repository}}:{{.Tag}}"])
        results: list[dict[str, Any]] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            image_id, tag = line.split("\t", 1)
            if name_filter and name_filter not in tag:
                continue
            results.append({"id": image_id, "tags": [tag], "size": None})
        return results

    def remove_image(self, image_ref: str) -> None:
        self._docker(["rmi", "-f", image_ref])


register("execution", SSHDockerExecutionTarget.name, SSHDockerExecutionTarget)
