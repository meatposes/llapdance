from llapdance.core.result import RunResult
from llapdance.plugins.storage.flat_file import FlatFileStorage


def test_flat_file_write_and_previous_for(tmp_path):
    storage = FlatFileStorage({"flat_file_dir": str(tmp_path)})

    first = RunResult(
        backend_name="engine-a",
        backend_config={},
        execution_target={},
        device_target={},
    )
    storage.write(first)

    assert storage.previous_for("engine-a") == [first]
    assert storage.previous_for("engine-b") == []

    second = RunResult(
        backend_name="engine-a",
        backend_config={},
        execution_target={},
        device_target={},
    )
    second.timestamp = first.timestamp + 10
    storage.write(second)

    latest = storage.previous_for("engine-a", limit=2)
    assert latest[0].run_id == second.run_id
    assert latest[1].run_id == first.run_id
