#!/usr/bin/env python3
"""Build a self-contained notebook from the current source, without publishing it."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import textwrap
import zipfile

REFERENCES = {
    "MINI_TUTORIAL_URL": ("https://github.com/nutonomy/nuscenes-devkit/blob/master/"
                          "python-sdk/tutorials/nuscenes_tutorial.ipynb"),
    "AWS_REGISTRY_URL": ("https://github.com/awslabs/open-data-registry/blob/main/"
                         "datasets/motional-nuscenes.yaml"),
}


def cell(kind, source, identifier):
    value = {"cell_type": kind, "id": identifier, "metadata": {},
             "source": textwrap.dedent(source).strip() + "\n"}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def source_bundle(root: Path):
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, text=True).split("\0")
    hashes, stream = {}, io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(set(names) - {""}):
            if (name.startswith("notebooks/")
                    or name.lower().endswith((".zip", ".ipynb", ".mp4", ".webm", ".mov"))):
                continue
            path = root / name
            if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Source file is unavailable or outside repository: {name}")
            data = path.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
        bundle.writestr("SOURCE_MANIFEST.json", json.dumps(hashes, sort_keys=True))
    return stream.getvalue(), hashes


def cpu_cells(cells):
    """Keep review/storage stages and replace every GPU-dependent execution stage."""
    replacements = {
        "scope": cell("markdown", """
            # Driving: a CPU-only image-feature reference experiment

            **Start in Colab with Python 3.12 and a CPU runtime**

            1. Open https://colab.research.google.com/ on a computer, choose
               **Upload notebook**, and upload this `.ipynb`. Keep the CPU runtime.
            2. Run the first five code cells in order: setup, restore check,
               dependencies, data preparation, and six-view review. The setup
               cell contains embedded source code and needs no manual changes.
            3. Stop at the review interface and label the images using the rubric.
               Run cells individually: displaying images does not pause **Run all**.
               The training checks reject incomplete labels.
            4. Before pausing, run **Save partial review** and download the backup.
               Colab storage may be lost after a reset; use this same notebook
               to restore the backup in a later session.

            Startup checks Python 3.12. Keep the complete error message if a
            version check or download fails. The project source is embedded;
            a separate GitHub checkout is unnecessary.

            The workflow prepares nuScenes mini, saves human six-view review,
            extracts frozen ResNet18 features and compares logistic regression,
            a small MLP, and matched SAEs with and without reward on CPU.

            The representation concatenates six ordered 512-dimensional image
            features into 3072 dimensions. It is a separate image-only reference
            protocol, separate from Cosmos hidden states and VLM results.
            The official pretrained checkpoint is downloaded and hash-verified;
            no random weights are substituted. This notebook starts a fresh run;
            it does not automatically reproduce the repository's saved results.

            Use a runtime whose normal network policy permits official data and
            PyTorch model downloads. Review the data terms before acquisition.
            Mini's frozen 6/2/2 scene split is a custom experiment allocation;
            mini has no official benchmark test split.

            Keep WORK on retained storage. Images, reviewed labels and feature
            artifacts are not embedded in this notebook. The portable workflow
            has synthetic fixture checks; real-image result records are separate.
        """, "scope"),
        "protocol": cell("code", """
            import importlib.metadata
            SEED, REVIEW_K, EPOCHS, HIDDEN_DIM, TOP_K = 42, 10, 10, 256, 32
            LAYER = 1
            protocol = dict(seed=SEED, review_k=REVIEW_K, epochs=EPOCHS,
                            hidden_dim=HIDDEN_DIM, k=TOP_K, layer=LAYER,
                            feature_protocol=FEATURE_PROTOCOL, pooling="camera_concat")
            protocol_path = WORK / "requested_protocol.json"
            if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
                raise ValueError("Protocol changed; choose a fresh WORK before continuing")
            protocol_path.write_text(json.dumps(protocol, indent=2) + "\\n")
            if sys.version_info[:2] != (3, 12):
                raise RuntimeError("The verified CPU lock requires a Python 3.12 runtime")

            def installed_versions():
                return sorted({f"{d.metadata['Name']}=={d.version}"
                               for d in importlib.metadata.distributions() if d.metadata['Name']})

            versions_path = WORK / "cpu_package_versions.txt"
            if versions_path.exists():
                if versions_path.read_text() != "\\n".join(installed_versions()) + "\\n":
                    raise RuntimeError("Restore recorded CPU packages before resuming this cache")
            else:
                run([sys.executable, "-m", "pip", "install", "-r",
                     ROOT/"requirements-cpu-lock.txt", "ipywidgets"])
                versions_path.write_text("\\n".join(installed_versions()) + "\\n")
        """, "protocol"),
        "gpu-text": cell("markdown", """
            ## 5. Download the pinned CPU encoder

            This step downloads the official ImageNet1K ResNet18 checkpoint
            (about 47 MB), verifies its full SHA256, and preserves existing files.
            It installs no additional model library. Keep the recorded CPU
            dependencies and preprocessing fixed while completing this cache.
        """, "cpu-weights-text"),
        "gpu": cell("code", """
            frozen_review = json.loads((ANNOTATIONS/"import_metadata.json").read_text())
            reviewed_hash = hashlib.sha256((WORK/"reviewed.csv").read_bytes()).hexdigest()
            if (not frozen_review["ready_for_current_training_contract"]
                    or list(frozen_review["review_inputs_sha256"].values()) != [reviewed_hash]):
                raise RuntimeError("Complete and import the reviewed labels before extraction")
            if versions_path.read_text() != "\\n".join(installed_versions()) + "\\n":
                raise RuntimeError("CPU environment changed since preparation")
            run([sys.executable, ROOT/"scripts/acquire_cpu_encoder.py", "--output", WEIGHTS])
        """, "cpu-weights"),
        "extract-text": cell("markdown", """
            ## 6. Measure one sample, then complete the same CPU cache

            One sample contains six views. The frozen encoder performs real CPU
            inference with the verified pretrained weights. Use one writer and
            storage supporting atomic rename/file sync. Preserve the first smoke
            report, then complete all selected samples before training.

            Layer 1 and the legacy `*_mean.npy` name are storage compatibility
            aliases for this protocol's camera concatenation; they do not claim
            decoder-layer MLP or token-mean features.
        """, "extract-text"),
        "smoke": cell("code", """
            import shutil

            def extract_command(limit):
                return [sys.executable, ROOT/"scripts/extract_cpu.py",
                        "--prepared-dir", PREPARED, "--data-root", DATA,
                        "--weights", WEIGHTS, "--threads", 2, "--save-every", 1,
                        "--max-samples", limit, "--output-dir", EXTRACT]

            run(extract_command(1))
            smoke_report = WORK/"smoke_extraction_summary.json"
            if not smoke_report.exists():
                shutil.copy2(EXTRACT/"extraction_summary.json", smoke_report)
            print(json.loads(smoke_report.read_text()))
        """, "smoke"),
        "complete-cache": cell("code", """
            if versions_path.read_text() != "\\n".join(installed_versions()) + "\\n":
                raise RuntimeError("CPU environment changed since smoke")
            run(extract_command(0))
            summary = json.loads((EXTRACT/"extraction_summary.json").read_text())
            if not summary.get("selection_complete"):
                raise RuntimeError("Extraction is partial; training remains blocked")
            run([sys.executable, ROOT/"scripts/run_real_driving.py", "validate",
                 "--prepared-dir", PREPARED, "--annotations-dir", ANNOTATIONS,
                 "--extract-dir", EXTRACT, "--layer", LAYER, "--pooling", "mean",
                 "--feature-protocol", FEATURE_PROTOCOL,
                 "--output", WORK/"handoff_validation.json"])
        """, "complete-cache"),
        "train": cell("code", """
            if versions_path.read_text() != "\\n".join(installed_versions()) + "\\n":
                raise RuntimeError("CPU environment changed since extraction")
            RUNS = WORK / "comparison_runs"
            run([sys.executable, ROOT/"scripts/run_real_driving.py", "train",
                 "--prepared-dir", PREPARED, "--annotations-dir", ANNOTATIONS,
                 "--extract-dir", EXTRACT, "--output-dir", RUNS, "--layer", LAYER,
                 "--pooling", "mean", "--feature-protocol", FEATURE_PROTOCOL,
                 "--review-k", REVIEW_K, "--seed", SEED, "--epochs", EPOCHS,
                 "--hidden-dim", HIDDEN_DIM, "--k", TOP_K])
            print((RUNS/"comparison/main_table.csv").read_text())
        """, "train"),
    }
    result = []
    titles = {
        "bootstrap": "Set up the project (leave embedded code unchanged)",
        "restore": "Restore a review backup (skip on the first run)",
        "protocol": "Install CPU dependencies and freeze settings",
        "acquire": "Download mini and prepare six-camera views",
        "review": "Open manual review (pause here to label images)",
    }
    for item in cells:
        item = replacements.get(item["id"], item)
        if item["id"] == "bootstrap":
            item = dict(item)
            item["source"] = item["source"].replace("driving-work", "driving-cpu-work").replace(
                'MODEL = WORK / "models/Cosmos-Reason1-7B"',
                'FEATURE_PROTOCOL = "cpu-resnet18"\n'
                'WEIGHTS = WORK / "models/resnet18/resnet18-f37072fd.pth"')
        if item["id"] in titles:
            item = dict(item)
            item["source"] = "#@title " + titles[item["id"]] + "\n" + item["source"]
            item["metadata"] = {**item["metadata"], "cellView": "form"}
        result.append(item)
        if item["id"] == "review":
            result.extend([
                cell("markdown", """
                    ### Save partial review before pausing

                    Review can span multiple sessions. This cell exports saved
                    decisions, frozen splits and settings, including incomplete
                    review. Images and models are excluded. In a later session,
                    open this same notebook, upload the backup and set its path
                    in the restore cell above.
                """, "partial-review-text"),
                cell("code", """
                    #@title Save partial review and download a small backup
                    if not (WORK/"reviewed.csv").is_file():
                        print("Save a review decision before running this cell.")
                    else:
                        from portable_handoff import pack_review
                        partial_backup = WORK/"partial_review_handoff.zip"
                        pack_review(WORK, partial_backup, SOURCE_SHA256)
                        try:
                            from google.colab import files as colab_files
                        except ImportError:
                            print("Save this file using your file browser:", partial_backup)
                        else:
                            colab_files.download(str(partial_backup))
                """, "partial-review-backup"),
            ])
    return result


def build(root: Path, destination: Path, feature_protocol="cosmos"):
    if feature_protocol not in ("cosmos", "cpu-resnet18"):
        raise ValueError("Unknown notebook feature protocol")
    payload, hashes = source_bundle(root)
    encoded = base64.b64encode(payload).decode("ascii")
    fingerprint = hashlib.sha256(payload).hexdigest()
    cells = [cell("markdown", """
        # Driving: real mini data to a controlled comparison

        This notebook contains the project source. It can run in Jupyter or a
        suitable Colab runtime without a GitHub upload or source-repository login.
        Its steps are acquisition, CPU review, one pinned GPU extraction, then
        four CPU models and the existing explorer.

        Source bootstrap, archive handling, review persistence and CPU comparisons
        have fixture checks. A complete Cosmos GPU run has not been verified in
        the managed workspace. The repository's saved ResNet18 CPU results use a
        separate feature protocol.

        Use an environment whose own network policy permits the official data
        and model hosts. The cells preserve inherited proxies and TLS checks.
        Review data and model terms at their official sources. Mini has ten
        scenes and no official test split; this workflow freezes a custom
        6/2/2 scene allocation.

        Keep WORK on storage you retain across runtime restarts. Review decisions
        save there immediately. Preserve the whole WORK directory for transfer.
    """, "scope")]
    bootstrap = f'''
import base64, hashlib, io, json, os, shlex, subprocess, sys, zipfile
from pathlib import Path, PurePosixPath

SOURCE_PAYLOAD = {encoded!r}
SOURCE_SHA256 = {fingerprint!r}
WORK = Path(os.environ.get("DRIVING_WORK_DIR",
    "/content/driving-work" if Path("/content").is_dir() else "driving-work")).resolve()
ROOT = WORK / "source"
WORK.mkdir(parents=True, exist_ok=True)
raw_source = base64.b64decode(SOURCE_PAYLOAD)
assert hashlib.sha256(raw_source).hexdigest() == SOURCE_SHA256
with zipfile.ZipFile(io.BytesIO(raw_source)) as bundle:
    source_manifest = json.loads(bundle.read("SOURCE_MANIFEST.json"))
    for name, expected in source_manifest.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source path")
        data = bundle.read(name)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Embedded source hash mismatch: " + name)
        path = ROOT / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Existing source differs; preserve it and choose a fresh WORK")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
sys.path.insert(0, str(ROOT / "scripts"))
DATA = Path(os.environ.get("DRIVING_DATA_ROOT", str(WORK / "dataset/nuscenes")))
PREPARED, ANNOTATIONS, EXTRACT = WORK/"prepared", WORK/"annotations", WORK/"extract"
MODEL = WORK / "models/Cosmos-Reason1-7B"

def run(args):
    args = [str(arg) for arg in args]
    print(shlex.join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)

print("Workspace:", WORK)
print("Verified source files:", len(source_manifest))
'''
    cells.append(cell("code", bootstrap, "bootstrap"))
    cells += [cell("markdown", """
        ## 1. Fix the proof-of-concept settings before seeing results

        Keep these settings for all four models. A changed protocol should use
        a new WORK folder. Do not choose another split after seeing test results.
    """, "protocol-text"), cell("code", """
        SEED, REVIEW_K, EPOCHS, HIDDEN_DIM, TOP_K = 42, 10, 10, 256, 32
        LAYER, MAX_PIXELS = 28, 1_600_000
        protocol = dict(seed=SEED, review_k=REVIEW_K, epochs=EPOCHS,
                        hidden_dim=HIDDEN_DIM, k=TOP_K, layer=LAYER, max_pixels=MAX_PIXELS)
        protocol_path = WORK / "requested_protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Protocol changed; choose a fresh WORK before continuing")
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\\n")
        # CPU data preparation needs no torch or CUDA installation.
        run([sys.executable, "-m", "pip", "install", "Pillow", "ipywidgets"])
    """, "protocol"), cell("markdown", """
        ## 2. Acquire mini, or use an already extracted local copy

        The [official tutorial][mini-tutorial]
        publishes the mini archive. Its Motional CDN is listed in the
        [AWS registry][aws-registry].
        Follow the [data terms](https://www.nuscenes.org/terms-of-use).

        The downloader records received-byte SHA256, checks transfer completeness
        and stages metadata/keyframe camera extraction. It omits sweeps and other
        sensors. A local hash is not an independently verified official checksum.
        Do not overwrite an existing data folder; preparation validates its files.

        [mini-tutorial]: MINI_TUTORIAL_URL
        [aws-registry]: AWS_REGISTRY_URL
    """, "acquire-text"), cell("code", """
        required_tables = ["scene", "sample", "sample_data", "calibrated_sensor", "sensor"]
        if all((DATA / "v1.0-mini" / (name + ".json")).is_file()
               for name in required_tables) and (DATA / "samples").is_dir():
            print("Using local data; preparation will validate references and selected images.")
        else:
            run([sys.executable, ROOT/"scripts/acquire_nuscenes_mini.py",
                 "--archive", WORK/"downloads/v1.0-mini.tgz", "--data-root", DATA,
                 "--report", WORK/"acquisition.json"])
        if not PREPARED.exists():
            run([sys.executable, ROOT/"scripts/prepare_driving.py", "prepare",
                 "--data-root", DATA, "--version", "v1.0-mini", "--frames-per-scene", 8,
                 "--seed", SEED, "--rubric-file", ROOT/"latest_grading_criteria.md",
                 "--output-dir", PREPARED])
        print(json.loads((PREPARED/"prepare_metadata.json").read_text()))
    """, "acquire"), cell("markdown", """
        ## 3. Review six views on CPU

        Enter your reviewer identifier. Each explicit label click saves the CSV
        atomically; rerunning this cell reloads it. Use the frozen rubric and keep
        uncertain cases uncertain. A model score is not a human label. Preserve
        independent review records and adjudicate before importing one final CSV.
    """, "review-text"), cell("code", """
        from notebook_review import ReviewSession
        review = ReviewSession(PREPARED, WORK/"reviewed.csv")
        print(review.progress())
        review.widget(DATA)
    """, "review"), cell("markdown", """
        ## 4. Freeze the reviewed labels

        Pending/uncertain decisions stay excluded. Each split needs both classes
        for this experiment contract. If it does not, keep this PoC as evidence of
        the limitation and plan more data; never fabricate labels or re-split by
        the test result. Use a new annotations directory for revised labels.
    """, "import-text"), cell("code", """
        if not ANNOTATIONS.exists():
            run([sys.executable, ROOT/"scripts/prepare_driving.py", "import-labels",
                 "--prepared-dir", PREPARED, "--reviews", WORK/"reviewed.csv",
                 "--output-dir", ANNOTATIONS])
        label_report = json.loads((ANNOTATIONS/"import_metadata.json").read_text())
        current_csv_hash = hashlib.sha256((WORK/"reviewed.csv").read_bytes()).hexdigest()
        if list(label_report["review_inputs_sha256"].values()) != [current_csv_hash]:
            raise ValueError("Reviews changed after import; preserve annotations and re-import")
        print(label_report["support"])
        if not label_report["ready_for_current_training_contract"]:
            raise ValueError("Insufficient reviewed class support; extraction/training blocked")
    """, "import"), cell("markdown", """
        ## 5. GPU prerequisites and pinned model

        This cell needs an existing CUDA runtime. NVIDIA's
        [upstream example](https://github.com/nvidia-cosmos/cosmos-reason1#inference)
        calls for a GPU with at least 24 GB. A free Colab GPU is not guaranteed;
        no paid upgrade is selected here. The guard below rejects small GPUs,
        but passing it does not establish that six full-resolution views fit.

        Follow the [Cosmos-Reason1-7B model terms](https://huggingface.co/nvidia/Cosmos-Reason1-7B).
        Resolve the official revision once and retain it. A changed pixel budget
        or dependency environment requires a fresh cache, not mixed features.
    """, "gpu-text"), cell("code", """
        frozen_review = json.loads((ANNOTATIONS/"import_metadata.json").read_text())
        reviewed_hash = hashlib.sha256((WORK/"reviewed.csv").read_bytes()).hexdigest()
        if (not frozen_review["ready_for_current_training_contract"]
                or list(frozen_review["review_inputs_sha256"].values()) != [reviewed_hash]):
            raise RuntimeError("Complete and import the reviewed labels before GPU work")
        import importlib.metadata, re, shutil

        def installed_versions():
            return sorted({f"{d.metadata['Name']}=={d.version}"
                           for d in importlib.metadata.distributions() if d.metadata['Name']})
        versions_path = WORK / "gpu_package_versions.txt"
        if versions_path.exists():
            if versions_path.read_text() != "\\n".join(installed_versions()) + "\\n":
                raise RuntimeError("Restore recorded package versions before resuming this cache")
        else:
            run([sys.executable, "-m", "pip", "install", "-r", ROOT/"requirements-gpu.txt"])
            versions_path.write_text("\\n".join(installed_versions()) + "\\n")

        # Probe in a fresh process, so pip changes cannot leave stale kernel imports.
        probe_code = ("import json,torch; "
                      "available=torch.cuda.is_available(); "
                      "p=torch.cuda.get_device_properties(0) if available else None; "
                      "print(json.dumps(dict(available=available, "
                      "name=p.name if p else None, memory=p.total_memory if p else 0, "
                      "bf16=torch.cuda.is_bf16_supported() if available else False)))")
        probe = subprocess.run([sys.executable, "-c", probe_code],
                               check=True, capture_output=True, text=True)
        gpu = json.loads(probe.stdout)
        print(gpu)
        if not gpu["available"]:
            raise RuntimeError("CUDA is absent. CPU preparation/review remain saved in WORK.")
        if gpu["memory"] < 24_000_000_000:
            raise RuntimeError("This guard requires at least 24 billion usable GPU bytes")
        DTYPE = "bfloat16" if gpu["bf16"] else "float16"
        from huggingface_hub import HfApi, snapshot_download
        revision_path = WORK / "model_revision.json"
        if revision_path.exists():
            MODEL_REVISION = json.loads(revision_path.read_text())["revision"]
        else:
            MODEL_REVISION = HfApi().model_info("nvidia/Cosmos-Reason1-7B").sha
            if not re.fullmatch(r"[0-9a-fA-F]{40}", MODEL_REVISION):
                raise ValueError("Model revision is not immutable")
            revision_path.write_text(json.dumps({"model": "nvidia/Cosmos-Reason1-7B",
                                                 "revision": MODEL_REVISION}, indent=2))
        info = HfApi().model_info("nvidia/Cosmos-Reason1-7B", revision=MODEL_REVISION,
                                 files_metadata=True)
        model_bytes = sum(item.size or 0 for item in info.siblings)
        if (not (MODEL/"config.json").exists()
                and shutil.disk_usage(WORK).free < model_bytes + 4*2**30):
            raise RuntimeError("Insufficient disk for pinned model plus extraction artifacts")
        snapshot_download("nvidia/Cosmos-Reason1-7B", revision=MODEL_REVISION, local_dir=MODEL)

        run([sys.executable, ROOT/"scripts/driving_preflight.py", "--data-root", DATA,
             "--version", "v1.0-mini", "--model-dir", MODEL,
             "--require", "all", "--output", WORK/"gpu_preflight.json"])
    """, "gpu"), cell("markdown", """
        ## 6. Measure one frame, then finish the same cache

        Use one writer and storage supporting atomic rename/file sync. The cache
        retains current/previous generations. Inspect smoke timing and peak VRAM.
        An OOM requires a new cache with an explicitly lower fixed pixel budget.
        Never train the one-frame smoke cache.
    """, "extract-text"), cell("code", """
        def extract_command(limit):
            return [sys.executable, ROOT/"scripts/extract.py",
                    "--samples-root", DATA/"samples", "--meta-dir", DATA/"v1.0-mini",
                    "--model-dir", MODEL, "--model-revision", MODEL_REVISION,
                    "--sample-tokens", PREPARED/"selected_samples.json", "--layers", LAYER,
                    "--dtype", DTYPE, "--min-pixels", 3136, "--max-pixels", MAX_PIXELS,
                    "--save-every", 1, "--max-samples", limit, "--output-dir", EXTRACT]
        run(extract_command(1))
        smoke_report = WORK/"smoke_extraction_summary.json"
        if not smoke_report.exists():
            shutil.copy2(EXTRACT/"extraction_summary.json", smoke_report)
        print(json.loads(smoke_report.read_text()))
    """, "smoke"), cell("code", """
        if versions_path.read_text() != "\\n".join(installed_versions()) + "\\n":
            raise RuntimeError("Environment changed since smoke")
        run(extract_command(0))
        summary = json.loads((EXTRACT/"extraction_summary.json").read_text())
        if not summary.get("selection_complete"):
            raise RuntimeError("Extraction is partial; training remains blocked")
        run([sys.executable, ROOT/"scripts/run_real_driving.py", "validate",
             "--prepared-dir", PREPARED, "--annotations-dir", ANNOTATIONS,
             "--extract-dir", EXTRACT, "--layer", LAYER,
             "--output", WORK/"handoff_validation.json"])
    """, "complete-cache"), cell("markdown", """
        ## 7. Run the controlled CPU comparison

        This launches Logistic regression, a small MLP and matched SAE with and
        without tail reward. It reuses the imported manifest and checks saved
        evidence before training; it never re-splits scenes. Start with one seed.
        Preserve exact commands, per-sample predictions and failures. A mini-only
        table remains a custom PoC even if the scores look good.
    """, "train-text"), cell("code", """
        # Reuse installed torch when continuing on GPU; CPU-only hosts may install it here.
        run([sys.executable, "-m", "pip", "install", "torch>=2.7.1", "numpy",
             "scikit-learn", "matplotlib", "Pillow"])
        RUNS = WORK / "comparison_runs"
        run([sys.executable, ROOT/"scripts/run_real_driving.py", "train",
             "--prepared-dir", PREPARED, "--annotations-dir", ANNOTATIONS,
             "--extract-dir", EXTRACT, "--output-dir", RUNS, "--layer", LAYER,
             "--review-k", REVIEW_K, "--seed", SEED, "--epochs", EPOCHS,
             "--hidden-dim", HIDDEN_DIM, "--k", TOP_K])
        print((RUNS/"comparison/main_table.csv").read_text())
    """, "train"), cell("markdown", """
        ## 8. Inspect and retain results

        For the browser, run the command below in a terminal that can reach its
        localhost server. A notebook output cell does not provide hosted access
        to that server. Keep its trained activation cache and authorized images.
        Include real FP/FN cases in the final demonstration; random weights and
        synthetic examples are not model-effectiveness evidence.

        Copy WORK using a transfer method permitted by your environment. Include
        hidden `.checkpoints/`, model revision, package versions, frozen prepared
        records, review CSV, annotations, extraction and comparison outputs.
        The source bundle contains no real data or weights and may be shared
        separately. Respect data/model redistribution conditions.
    """, "retain-text"), cell("code", """
        browser_command = [sys.executable, ROOT/"scripts/neuron_explorer.py", "serve",
                           "--run", RUNS/"sae_abstopk_norm", "--samples-root", DATA/"samples"]
        print(shlex.join([str(arg) for arg in browser_command]))
        print("Keep the complete work directory:", WORK)
    """, "retain")]
    cells[2:2] = [cell("markdown", """
        ## Storage and optional review restore

        Colab's `/content` storage is ephemeral: switching runtime can remove it.
        Use retained storage via DRIVING_WORK_DIR before bootstrap, or download
        the small review backup cell below before switching. Keep this same
        notebook to restore that backup; its source fingerprint is checked.
        Images and weights are not included in the review backup. The acquisition
        cell can reacquire data while preserving the restored split and decisions.
    """, "restore-text"), cell("code", """
        # After uploading a saved review_handoff.zip to a new runtime, set its path here.
        REVIEW_BACKUP_TO_RESTORE = None
        if REVIEW_BACKUP_TO_RESTORE is not None:
            from portable_handoff import restore_review
            restored = restore_review(Path(REVIEW_BACKUP_TO_RESTORE), WORK, SOURCE_SHA256)
            print("Restored review files:", len(restored["files_sha256"]))
    """, "restore")]
    gpu_index = next(i for i, item in enumerate(cells) if item["id"] == "gpu-text")
    cells[gpu_index:gpu_index] = [cell("markdown", """
        ## Save review progress before changing runtimes

        This creates a small backup of prepared records, review CSV, annotations
        and the chosen protocol. On Colab it offers a browser download; in other
        Jupyter environments retain the printed ZIP using your file browser.
        Re-run this cell after recording model/version information to include it.
        Preserve feature caches and authorized images separately after extraction.
    """, "backup-text"), cell("code", """
        from portable_handoff import pack_review
        backup_path = WORK / "review_handoff.zip"
        pack_review(WORK, backup_path, SOURCE_SHA256)
        print("Review backup:", backup_path, "bytes:", backup_path.stat().st_size)
        try:
            from google.colab import files as colab_files
        except ImportError:
            print("Retain this ZIP before replacing the runtime or workspace.")
        else:
            colab_files.download(str(backup_path))
    """, "backup")]
    if feature_protocol == "cpu-resnet18":
        cells = cpu_cells(cells)
    for item in cells:
        if item["cell_type"] == "markdown":
            for marker, url in REFERENCES.items():
                item["source"] = item["source"].replace(marker, url)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"},
                             "driving_source_sha256": fingerprint,
                             "feature_protocol": feature_protocol,
                             "source_files": len(hashes),
                             "verification_scope": (
                                 "CPU fixture checks; saved real-image results "
                                 "recorded separately" if
                                 feature_protocol == "cpu-resnet18" else
                                 "CPU fixture checks; Cosmos GPU execution unverified")}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return {"path": str(destination), "source_files": len(hashes),
            "source_sha256": fingerprint, "bytes": destination.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--feature-protocol", choices=("cosmos", "cpu-resnet18"),
                        default="cosmos")
    args = parser.parse_args()
    destination = args.output or Path("notebooks/" + (
        "driving_cpu_from_zero.ipynb" if args.feature_protocol == "cpu-resnet18"
        else "driving_from_zero.ipynb"))
    print(json.dumps(build(Path(__file__).resolve().parents[1], destination,
                           args.feature_protocol), indent=2))


if __name__ == "__main__":
    main()
