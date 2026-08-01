"""Local docker-socket execution target (SPEC.md §5, §6). Reference
implementation of ExecutionTargetAdapter - a remote SSH-based adapter
implements the same interface (SPEC.md §15.3, not yet built)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import docker
from docker.models.containers import Container

from llapdance.plugins.base import ExecutionTargetAdapter, RunningBackend
from llapdance.plugins.registry import register


class DockerRunningBackend(RunningBackend):
    def __init__(self, container: Container, endpoint: str) -> None:
        self._container = container
        self._endpoint = endpoint

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def container(self) -> Container:
        return self._container

    def logs(self, tail: int = 200) -> str:
        return self._container.logs(tail=tail).decode("utf-8", errors="replace")


class LocalDockerExecutionTarget(ExecutionTargetAdapter):
    name = "local-docker"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # config accepted for a uniform `cls(config)` instantiation
        # convention across adapter kinds; local docker needs none of it.
        self._client = docker.from_env()

    def build(self, backend_config: dict[str, Any]) -> str:
        source = backend_config["source"]
        if source["mode"] == "prebuilt":
            image = source["image"]
            try:
                self._client.images.get(image)
            except docker.errors.ImageNotFound:
                self._client.images.pull(image)
            return image

        build = source["build"]
        clone_path = Path(build["path"])
        if not clone_path.exists():
            clone_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--branch", build["ref"], build["repo"], str(clone_path)],
                check=True,
            )
        else:
            subprocess.run(["git", "-C", str(clone_path), "fetch", "origin", build["ref"]], check=True)
            subprocess.run(["git", "-C", str(clone_path), "checkout", build["ref"]], check=True)

        image_tag = f"llapdance/{backend_config['name']}:{build['ref']}"
        image, _logs = self._client.images.build(
            path=str(clone_path),
            dockerfile=build["dockerfile"],
            tag=image_tag,
            buildargs=build["build_args"],
            rm=True,
        )
        return image.tags[0] if image.tags else image_tag

    def start(self, backend_config: dict[str, Any], image_ref: str, device_indices: list[int]) -> RunningBackend:
        network_cfg = backend_config.get("network", {})
        network_mode = network_cfg.get("mode", "disabled")
        network_name = network_cfg.get("network") if network_mode == "enabled" else None

        env = dict(backend_config.get("env", {}))
        if device_indices:
            # Generic device-selection hint; vendor-specific pinning env vars
            # are layered on by whichever GPU adapter resolves this list
            # (SPEC.md §7) - not hardcoded to one vendor's convention here.
            env["LLAPDANCE_DEVICE_INDICES"] = ",".join(str(i) for i in device_indices)

        port = backend_config.get("port", 8000)
        volumes = {
            host: {"bind": container_path, "mode": "ro"}
            for host, container_path in backend_config.get("volumes", {}).items()
        }
        container = self._client.containers.run(
            image_ref,
            command=backend_config.get("command") or None,
            detach=True,
            environment=env,
            volumes=volumes or None,
            devices=backend_config.get("devices") or None,
            ports={f"{port}/tcp": None} if network_mode != "isolated" else None,
            network=network_name,
            network_disabled=(network_mode == "isolated"),
        )
        container.reload()
        if network_mode == "isolated":
            endpoint = f"http://{container.attrs['NetworkSettings']['IPAddress']}:{port}"
        else:
            host_port = container.ports.get(f"{port}/tcp", [{}])[0].get("HostPort", port)
            endpoint = f"http://127.0.0.1:{host_port}"
        return DockerRunningBackend(container, endpoint)

    def stop(self, backend: RunningBackend) -> None:
        assert isinstance(backend, DockerRunningBackend)
        backend.container.stop(timeout=10)
        backend.container.remove(force=True)

    def list_images(self, name_filter: str | None = None) -> list[dict[str, Any]]:
        images = self._client.images.list()
        results = []
        for img in images:
            tags = img.tags
            if name_filter and not any(name_filter in t for t in tags):
                continue
            results.append({"id": img.id, "tags": tags, "size": img.attrs.get("Size")})
        return results

    def remove_image(self, image_ref: str) -> None:
        self._client.images.remove(image_ref, force=True)


register("execution", LocalDockerExecutionTarget.name, LocalDockerExecutionTarget)
