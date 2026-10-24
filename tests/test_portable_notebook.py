"""Execute portable bootstrap/preparation with synthetic local data; no GPU or download."""

import ast
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_driving_notebook import build  # noqa: E402
from test_prepare_driving import synthetic_nuscenes  # noqa: E402


def notebook(tmp_path, feature_protocol="cosmos"):
    path = tmp_path / "driving.ipynb"
    report = build(ROOT, path, feature_protocol)
    content = json.loads(path.read_text())
    cells = {row["id"]: row["source"] for row in content["cells"]}
    return report, content, cells


def test_all_notebook_code_compiles_and_bootstrap_verifies_sources(tmp_path, monkeypatch):
    report, content, cells = notebook(tmp_path)
    for row in content["cells"]:
        if row["cell_type"] == "code":
            ast.parse(row["source"])
            assert row["execution_count"] is None and not row["outputs"]
    monkeypatch.setenv("DRIVING_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {}
    exec(cells["bootstrap"], namespace)
    assert namespace["SOURCE_SHA256"] == report["source_sha256"]
    assert (namespace["ROOT"] / "scripts/run_real_driving.py").is_file()
    # Preserve edits rather than silently overwriting a user's existing source.
    target = namespace["ROOT"] / "scripts/prepare_driving.py"
    target.write_text("user-modified source")
    with pytest.raises(ValueError, match="Existing source differs"):
        exec(cells["bootstrap"], {})
    assert target.read_text() == "user-modified source"


def test_notebook_local_fixture_runs_actual_preparation_without_downloading(tmp_path, monkeypatch):
    _, _, cells = notebook(tmp_path)
    data = synthetic_nuscenes(tmp_path / "synthetic-data")
    monkeypatch.setenv("DRIVING_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("DRIVING_DATA_ROOT", str(data))
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {"SEED": 42}
    exec(cells["bootstrap"], namespace)
    exec(cells["acquire"], namespace)
    prepared = namespace["PREPARED"]
    metadata = json.loads((prepared / "prepare_metadata.json").read_text())
    assert metadata["scene_counts"] == {"train": 6, "val": 2, "test": 2}
    assert metadata["labels_created"] is False
    assert not (namespace["WORK"] / "downloads").exists()


def test_cpu_notebook_bootstraps_separate_work_and_has_no_gpu_stages(tmp_path, monkeypatch):
    _, content, cells = notebook(tmp_path, "cpu-resnet18")
    assert content["metadata"]["feature_protocol"] == "cpu-resnet18"
    assert "gpu" not in cells and "gpu-text" not in cells
    assert {"cpu-weights", "review", "import", "backup", "restore"} <= cells.keys()
    assert "partial-review-backup" in cells
    code_ids = [row["id"] for row in content["cells"] if row["cell_type"] == "code"]
    assert code_ids[:5] == ["bootstrap", "restore", "protocol", "acquire", "review"]
    assert code_ids.index("review") < code_ids.index("partial-review-backup") < code_ids.index(
        "import")
    for row in content["cells"]:
        if row["cell_type"] == "code":
            ast.parse(row["source"])
            assert "huggingface_hub" not in row["source"]
            assert "requirements-gpu.txt" not in row["source"]
            assert "torch.cuda" not in row["source"]
    monkeypatch.delenv("DRIVING_WORK_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {}
    exec(cells["bootstrap"], namespace)
    assert namespace["WORK"].name == "driving-cpu-work"
    assert namespace["FEATURE_PROTOCOL"] == "cpu-resnet18"
    assert namespace["WEIGHTS"].name == "resnet18-f37072fd.pth"
    assert (namespace["ROOT"] / "scripts/extract_cpu.py").is_file()
    assert (namespace["ROOT"] / "scripts/acquire_cpu_encoder.py").is_file()


def test_cpu_notebook_prepares_local_fixture_without_download_or_accelerator(tmp_path, monkeypatch):
    _, _, cells = notebook(tmp_path, "cpu-resnet18")
    data = synthetic_nuscenes(tmp_path / "synthetic-data")
    monkeypatch.setenv("DRIVING_WORK_DIR", str(tmp_path / "cpu-work"))
    monkeypatch.setenv("DRIVING_DATA_ROOT", str(data))
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {"SEED": 42}
    exec(cells["bootstrap"], namespace)
    exec(cells["acquire"], namespace)
    report = json.loads((namespace["PREPARED"] / "prepare_metadata.json").read_text())
    assert report["scene_counts"] == {"train": 6, "val": 2, "test": 2}
    assert report["labels_created"] is False
    assert not (namespace["WORK"] / "downloads").exists()


def test_cpu_notebook_forwards_its_protocol_to_extraction_validation_and_training(
        tmp_path, monkeypatch):
    _, _, cells = notebook(tmp_path, "cpu-resnet18")
    monkeypatch.setenv("DRIVING_WORK_DIR", str(tmp_path / "cpu-work"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {}
    exec(cells["bootstrap"], namespace)
    extract = namespace["EXTRACT"]
    extract.mkdir()
    (extract / "extraction_summary.json").write_text('{"selection_complete": true}\n')
    versions = namespace["WORK"] / "test-only-package-versions.txt"
    versions.write_text("synthetic-package-record\n")
    commands = []

    def record_command(args):
        commands.append([str(arg) for arg in args])
        if "train" in commands[-1]:
            table = namespace["WORK"] / "comparison_runs/comparison/main_table.csv"
            table.parent.mkdir(parents=True)
            table.write_text("SYNTHETIC command-forwarding test only\n")

    namespace.update(run=record_command, versions_path=versions,
                     installed_versions=lambda: ["synthetic-package-record"],
                     LAYER=1, SEED=42, REVIEW_K=10, EPOCHS=1, HIDDEN_DIM=8, TOP_K=2)
    exec(cells["smoke"], namespace)
    exec(cells["complete-cache"], namespace)
    exec(cells["train"], namespace)
    extraction = [args for args in commands if any(arg.endswith("extract_cpu.py") for arg in args)]
    assert [args[args.index("--max-samples") + 1] for args in extraction] == ["1", "0"]
    for args in commands:
        if any(arg.endswith("run_real_driving.py") for arg in args):
            assert args[args.index("--feature-protocol") + 1] == "cpu-resnet18"
            assert args[args.index("--layer") + 1] == "1"
            assert args[args.index("--pooling") + 1] == "mean"
    assert any("train" in args for args in commands)


def test_notebook_rejects_unknown_feature_protocol(tmp_path):
    with pytest.raises(ValueError, match="Unknown notebook feature protocol"):
        build(ROOT, tmp_path / "unknown.ipynb", "untracked-model")
