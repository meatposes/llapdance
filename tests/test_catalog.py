import json

import pytest

from llapdance.core.catalog import LabeledImageRemovalError, get_labels, label_image, list_images, remove_image


class FakeExecution:
    def __init__(self, images):
        self._images = images
        self.removed = []

    def list_images(self, name_filter=None):
        if name_filter:
            return [i for i in self._images if any(name_filter in t for t in i["tags"])]
        return self._images

    def remove_image(self, image_ref):
        self.removed.append(image_ref)


def test_label_image_persists_and_is_readable_back(tmp_path):
    label_image(str(tmp_path), "qxmx:latest", "good", note="fast and correct")
    labels = get_labels(str(tmp_path))
    assert labels["qxmx:latest"]["label"] == "good"
    assert labels["qxmx:latest"]["note"] == "fast and correct"
    assert "labeled_at" in labels["qxmx:latest"]


def test_label_image_overwrites_previous_label(tmp_path):
    label_image(str(tmp_path), "qxmx:latest", "good")
    label_image(str(tmp_path), "qxmx:latest", "bad", note="regressed")
    labels = get_labels(str(tmp_path))
    assert labels["qxmx:latest"]["label"] == "bad"


def test_list_images_enriches_with_label_and_runs(tmp_path):
    (tmp_path / "1_engine-a_abc123.json").write_text(
        json.dumps({"run_id": "abc123", "backend_name": "engine-a", "timestamp": 1.0, "image_ref": "qxmx:latest"})
    )
    label_image(str(tmp_path), "qxmx:latest", "good")

    execution = FakeExecution([{"id": "sha256:1", "tags": ["qxmx:latest"], "size": 123}])
    result = list_images(execution, catalog_dir=str(tmp_path))

    assert len(result) == 1
    assert result[0]["label"]["label"] == "good"
    assert result[0]["runs"] == [{"run_id": "abc123", "backend_name": "engine-a", "timestamp": 1.0}]


def test_list_images_without_catalog_dir_has_no_label_or_runs():
    execution = FakeExecution([{"id": "sha256:1", "tags": ["untested:latest"], "size": 1}])
    result = list_images(execution)
    assert result[0]["label"] is None
    assert result[0]["runs"] == []


def test_list_images_ignores_the_labels_file_itself_when_indexing_runs(tmp_path):
    label_image(str(tmp_path), "qxmx:latest", "good")  # writes _image_labels.json
    execution = FakeExecution([{"id": "sha256:1", "tags": ["qxmx:latest"], "size": 1}])
    result = list_images(execution, catalog_dir=str(tmp_path))
    assert result[0]["runs"] == []  # labels file isn't mistaken for a run result


def test_remove_image_refuses_labeled_good_without_force(tmp_path):
    label_image(str(tmp_path), "qxmx:latest", "good")
    execution = FakeExecution([])
    with pytest.raises(LabeledImageRemovalError):
        remove_image(execution, "qxmx:latest", catalog_dir=str(tmp_path))
    assert execution.removed == []


def test_remove_image_allows_labeled_good_with_force(tmp_path):
    label_image(str(tmp_path), "qxmx:latest", "good")
    execution = FakeExecution([])
    remove_image(execution, "qxmx:latest", catalog_dir=str(tmp_path), force=True)
    assert execution.removed == ["qxmx:latest"]


def test_remove_image_allows_unlabeled_or_bad_without_force(tmp_path):
    label_image(str(tmp_path), "qxmx:old", "bad")
    execution = FakeExecution([])
    remove_image(execution, "qxmx:old", catalog_dir=str(tmp_path))
    remove_image(execution, "qxmx:never-labeled", catalog_dir=str(tmp_path))
    assert execution.removed == ["qxmx:old", "qxmx:never-labeled"]
