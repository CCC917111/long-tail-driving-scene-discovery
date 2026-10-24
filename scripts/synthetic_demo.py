#!/usr/bin/env python3
"""Run four small CPU experiments and the browser on conspicuously synthetic data.

No nuScenes assets, VLM, GPU or download is used. Both SAE checkpoints are
actually trained for the requested epochs; random initial weights are never
presented as a trained model. Scores from this fixture are software checks,
not evidence about driving-scene discovery.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tail_activations import CAMERAS


def create_dataset(output_dir: Path, *, scenes: int, seed: int,
                   namespace: str = "synthetic", include_labels: bool = True) -> dict:
    if scenes < 10:
        raise ValueError("Use at least 10 scenes to keep validation/test nonempty")
    rng = np.random.default_rng(seed)
    data_dir = output_dir / "dataset"
    images_dir = data_dir / "samples"
    for camera in CAMERAS:
        (images_dir / camera).mkdir(parents=True, exist_ok=True)
    features, meta, labels = [], [], []
    for scene in range(scenes):
        for frame in range(4):
            token = f"{namespace}-scene-{scene:03d}-frame-{frame}"
            label = frame % 2
            feature = rng.normal(0, 0.35, size=8)
            feature[0] += 2 * label - 1
            # Deliberate contradictory examples make failure inspection possible.
            if scene % 7 == 0 and frame in (0, 1):
                feature[0] *= -1
            features.append(feature)
            image_paths = {}
            for camera in CAMERAS:
                rel = Path(camera) / f"{token}.png"
                image_paths[camera] = rel.as_posix()
                image = Image.new("RGB", (360, 180), (35 + scene % 60, 50, 90))
                draw = ImageDraw.Draw(image)
                draw.rectangle((12, 12, 348, 168), outline=(240, 175, 30), width=3)
                draw.text((24, 27), "SYNTHETIC SOFTWARE FIXTURE", fill="white")
                draw.text((24, 60), "NO REAL DRIVING IMAGE / NO VLM", fill="white")
                draw.text((24, 96), token, fill="white")
                draw.text((24, 130), camera, fill="white")
                image.save(images_dir / rel)
            meta.append({"sample_token": token, "scene_token": f"{namespace}-scene-{scene:03d}",
                         "scene_name": f"SYNTHETIC scene {scene:03d}", "images": image_paths})
            labels.append({"sample_token": token,
                           "label": "long_tail" if label else "normal_core",
                           "label_source": "synthetic-generation-v1",
                           "review_status": "synthetic-not-human-reviewed"})
    np.save(data_dir / "features.npy", np.asarray(features, dtype=np.float32))
    provenance = {"kind": "synthetic-software-fixture", "seed": seed,
                  "feature_dim": 8, "scene_count": scenes, "frames_per_scene": 4,
                  "model_revision": "not-applicable-no-VLM-used", "layer": None,
                  "pooling": "not-applicable-generated-vectors", "prompt_version": None,
                  "rubric_version": "synthetic-binary-v1",
                  "warning": "These are generated vectors and images, not nuScenes data."}
    files = [("meta.json", meta), ("feature_config.json", provenance)]
    if include_labels:
        files.append(("labels.json", labels))
    for filename, value in files:
        (data_dir / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    (output_dir / "SYNTHETIC.txt").write_text(
        "SYNTHETIC SOFTWARE DEMONSTRATION ONLY.\n"
        "Generated vectors, labels and placeholder images; no nuScenes data or VLM.\n"
        "SAE weights in the run folders were trained on this fixture.\n"
        "Do not report fixture metrics as driving-model performance.\n", encoding="utf-8")
    return provenance


def verify_reloads(output_dir: Path) -> dict:
    """Load every saved checkpoint and independently reproduce saved scores."""
    import joblib
    import torch
    from evaluate_runs import file_sha256
    from screen import build_model, encode
    from tail_activations import tail_score

    features = np.load(output_dir / "dataset/features.npy")
    meta = json.loads((output_dir / "dataset/meta.json").read_text())
    manifest = json.loads((output_dir / "protocol/manifest.json").read_text())
    train_tokens = {m["sample_token"] for m in manifest if m["split"] == "train"}
    train = [i for i, row in enumerate(meta) if row["sample_token"] in train_tokens]
    expected_mean = features[train].mean(axis=0, keepdims=True)
    expected_std = features[train].std(axis=0, keepdims=True) + 1e-8
    report = {}
    for method in ("logistic", "mlp", "sae_none", "sae_norm"):
        run_dir = output_dir / method
        with np.load(run_dir / "standardizer.npz") as standardizer:
            mean, std = standardizer["mean"], standardizer["std"]
        np.testing.assert_array_equal(mean, expected_mean)
        np.testing.assert_array_equal(std, expected_std)
        x = ((features - mean) / std).astype(np.float32)
        if method.startswith("sae_"):
            config = json.loads((run_dir / "metrics.json").read_text())["config"]
            model = build_model(config, x.shape[1], "cpu")
            checkpoint = run_dir / "best_model.pth"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.eval()
            scores = tail_score(encode(model, x, 32, "cpu"))
            torch.manual_seed(config["seed"])
            initial = build_model(config, x.shape[1], "cpu")
            if all(torch.equal(value, initial.state_dict()[key])
                   for key, value in model.state_dict().items()):
                raise AssertionError(f"{method}: checkpoint is still the initial random model")
        else:
            checkpoint = run_dir / "model.joblib"
            saved = joblib.load(checkpoint)
            np.testing.assert_array_equal(saved["mean"], mean)
            np.testing.assert_array_equal(saved["std"], std)
            estimator = saved["estimator"]
            scores = estimator.predict_proba(x)[:, list(estimator.classes_).index(1)]
        predicted = json.loads((run_dir / "predictions.json").read_text())
        by_token = {row["sample_token"]: row for row in predicted}
        expected = np.array([by_token[row["sample_token"]]["score"] for row in meta])
        np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-6)
        report[method] = {"scores_match": True, "train_only_standardizer": True,
                          "checkpoint_sha256": file_sha256(checkpoint),
                          "max_score_difference": float(np.abs(scores - expected).max())}
    (output_dir / "reload_check.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_screening(output_dir: Path) -> dict:
    """Check new, unlabelled tokens against direct inference from the trained SAE."""
    import csv
    import torch
    from neuron_explorer import build_explorer, parse_args
    from screen import build_model, encode
    from tail_activations import ActivationStore, predict_long_tail, tail_score

    run_dir = output_dir / "sae_norm"
    screening = output_dir / "screening"
    new_data = output_dir / "new_samples/dataset"
    meta = json.loads((new_data / "meta.json").read_text())
    original = json.loads((output_dir / "dataset/meta.json").read_text())
    assert not ({r["sample_token"] for r in meta} & {r["sample_token"] for r in original})
    assert not (new_data / "labels.json").exists(), "Screening fixture must be unlabelled"
    features = np.load(new_data / "features.npy")
    with np.load(run_dir / "standardizer.npz") as normalizer:
        x = ((features - normalizer["mean"]) / normalizer["std"]).astype(np.float32)
    config = json.loads((run_dir / "metrics.json").read_text())["config"]
    model = build_model(config, x.shape[1], "cpu")
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location="cpu",
                                     weights_only=True))
    model.eval()
    z_t = encode(model, x, 32, "cpu")
    store = ActivationStore(screening)
    np.testing.assert_allclose(store.dense(), z_t, rtol=1e-6, atol=1e-6)
    explorer = build_explorer(parse_args([
        "units", "--run", str(screening), "--samples-root", str(new_data / "samples"),
        "--neuron-threshold", "0.01"]))
    assert explorer.eta == 0.05, "Screening override must survive browser reload"
    with (screening / "screening.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(meta) and store.labels is None
    by_token = {r["sample_token"]: r for r in rows}
    assert set(by_token) == {r["sample_token"] for r in meta}
    scores, predicted = tail_score(z_t), predict_long_tail(z_t, explorer.eta)
    for i, entry in enumerate(meta):
        token = entry["sample_token"]
        row, frame = by_token[token], explorer.units_of_frame(token)
        np.testing.assert_allclose(float(row["tail_score"]), scores[i], rtol=1e-6)
        expected = "long_tail" if predicted[i] else "normal"
        assert row["prediction"] == frame["prediction"] == expected
        assert len(frame["cameras"]) == 6
    report = {"independent_tokens": True, "requires_labels": False, "n_samples": len(rows),
              "activation_reload_matches": True, "effective_eta": explorer.eta,
              "browser_predictions_match": True,
              "checkpoint": str(run_dir / "best_model.pth")}
    (output_dir / "screening_check.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_stage(name: str, command: list[str], *, output_dir: Path, env: dict) -> None:
    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / f"{name}.txt").open("w", encoding="utf-8") as handle:
        handle.write(shlex.join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                cwd=Path(__file__).resolve().parents[1], check=False)
    if result.returncode:
        raise RuntimeError(f"Synthetic stage {name} failed; inspect {logs / (name + '.txt')}")
    print(f"[SYNTHETIC] {name}: passed", flush=True)


def verify_browser(output_dir: Path) -> dict:
    from neuron_explorer import Explorer, frame_sheet, unit_sheet
    from tail_activations import ActivationStore

    run_dir = output_dir / "sae_norm"
    rows = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    errors = [row for row in rows if row["split"] == "test" and
              row["prediction"] != row["label"]]
    if not errors:
        raise RuntimeError("This fixture produced no fixed-rule test error to inspect")
    explorer = Explorer(ActivationStore(run_dir), output_dir / "dataset/samples",
                        neuron_threshold=0.01)
    visible = {row["unit"] for row in explorer.units(top=80)}
    selected = None
    for row in errors:
        frame = explorer.units_of_frame(row["sample_token"])
        for entry in frame["units"]:
            unit = entry["unit"]
            if unit in visible and any(
                    item["sample_token"] == row["sample_token"]
                    for item in explorer.frames_of_unit(unit, top=48)):
                selected = row, frame, unit
                break
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError("No synthetic test error is reachable through a displayed unit")
    frame_row, frame, unit = selected
    browser_dir = output_dir / "browser"
    browser_dir.mkdir(exist_ok=True)
    frame_sheet(explorer, frame).save(browser_dir / "sample.png")
    assert len(frame["cameras"]) == 6, "All six synthetic image paths must be readable"
    if not frame["units"]:
        raise RuntimeError("Selected synthetic frame has no nonzero tail activation")
    neighbours = explorer.frames_of_unit(unit, top=6)
    if not neighbours:
        raise RuntimeError("No high-activation neighbours available for browser smoke test")
    unit_sheet(explorer, unit, neighbours, "CAM_FRONT").save(browser_dir / "unit.png")
    report = {"kind": "synthetic-browser-smoke", "sample_token": frame["sample_token"],
              "unit": unit, "fixed_rule_label": frame_row["label"],
              "fixed_rule_prediction": frame_row["prediction"],
              "error": "fp" if frame_row["prediction"] == 1 else "fn",
              "image_count": len(frame["cameras"]), "neighbour_count": len(neighbours),
              "reachable_from_ui_unit": True,
              "trained_checkpoint": str(run_dir / "best_model.pth")}
    (browser_dir / "smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=20)
    parser.add_argument("--http", action="store_true",
                        help="Also start real localhost HTTP servers and verify browser routes")
    args = parser.parse_args()
    if args.epochs < 1 or args.bootstrap_samples < 0:
        raise ValueError("epochs must be positive and bootstrap-samples nonnegative")
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh, empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    provenance = create_dataset(out, scenes=args.scenes, seed=args.seed)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MPLCONFIGDIR=str(out / ".matplotlib"), XDG_CACHE_HOME=str(out / ".cache"))
    scripts = Path(__file__).resolve().parent
    data = ["--features", str(out / "dataset/features.npy"),
            "--meta", str(out / "dataset/meta.json"),
            "--labels", str(out / "dataset/labels.json"),
            "--feature-config", str(out / "dataset/feature_config.json"),
            "--seed", str(args.seed), "--review-k", "8",
            "--bootstrap-samples", str(args.bootstrap_samples)]
    sae = [sys.executable, str(scripts / "sae_abstopk_tail_reward.py"), *data,
           "--hidden-dim", "16", "--k", "4", "--epochs", str(args.epochs),
           "--batch-size", "32", "--dropout", "0", "--device", "cpu"]
    commands = [("freeze", [*sae, "--output-dir", str(out / "protocol"),
                            "--freeze-manifest-only"])]
    manifest = ["--manifest", str(out / "protocol/manifest.json")]
    for method in ("logistic", "mlp"):
        command = [sys.executable, str(scripts / "supervised_baselines.py"), *data, *manifest,
                   "--method", method, "--output-dir", str(out / method),
                   "--epochs", str(args.epochs), "--hidden-dim", "16", "--batch-size", "32"]
        commands.append((method, command))
    for reward in ("none", "norm"):
        commands.append((f"sae_{reward}", [*sae, *manifest, "--reward", reward,
                                           "--output-dir", str(out / f"sae_{reward}")]))
    run_dirs = [out / name for name in ("logistic", "mlp", "sae_none", "sae_norm")]
    comparison = [sys.executable, str(scripts / "summarize_runs.py"), "--runs",
                  *map(str, run_dirs), "--output-dir", str(out / "comparison")]
    commands.append(("comparison", comparison))
    create_dataset(out / "new_samples", scenes=10, seed=args.seed + 1,
                   namespace="synthetic-unlabelled", include_labels=False)
    commands.append(("screen", [
        sys.executable, str(scripts / "screen.py"), "--run-dir", str(out / "sae_norm"),
        "--features", str(out / "new_samples/dataset/features.npy"),
        "--meta", str(out / "new_samples/dataset/meta.json"),
        "--output-dir", str(out / "screening"), "--eta", "0.05",
        "--neuron-threshold", "0.01", "--batch-size", "32", "--device", "cpu"]))
    for name, command in commands:
        run_stage(name, command, output_dir=out, env=env)
    reloads = verify_reloads(out)
    screening = verify_screening(out)
    browser = verify_browser(out)
    http = {"status": "not_requested", "reason": "Run with --http to verify real HTTP"}
    if args.http:
        from neuron_explorer import build_explorer, parse_args as explorer_args
        from verify_browser_http import verify_http

        http = {}
        for name, run_dir, samples_root, token in (
                ("trained", out / "sae_norm", out / "dataset/samples", browser["sample_token"]),
                ("screened", out / "screening", out / "new_samples/dataset/samples",
                 "synthetic-unlabelled-scene-000-frame-0")):
            ex = build_explorer(explorer_args([
                "units", "--run", str(run_dir), "--samples-root", str(samples_root),
                "--neuron-threshold", "0.01"]))
            frame = ex.units_of_frame(token)
            if not frame["units"]:
                raise AssertionError(f"{name}: no active unit to inspect")
            http[name] = verify_http(ex, out / "browser" / name, token,
                                     frame["units"][0]["unit"],
                                     checkpoint_path=out / "sae_norm/best_model.pth")
    launch = [sys.executable, str(scripts / "neuron_explorer.py"), "serve", "--run",
              str(out / "sae_norm"), "--samples-root", str(out / "dataset/samples"),
              "--neuron-threshold", "0.01"]
    (out / "browser_command.txt").write_text(shlex.join(launch) + "\n", encoding="utf-8")
    screen_launch = [sys.executable, str(scripts / "neuron_explorer.py"), "serve", "--run",
                     str(out / "screening"), "--samples-root",
                     str(out / "new_samples/dataset/samples"), "--neuron-threshold", "0.01"]
    (out / "screening_browser_command.txt").write_text(shlex.join(screen_launch) + "\n")
    (out / "commands.txt").write_text(
        "\n".join(shlex.join(command) for _, command in commands) + "\n", encoding="utf-8")
    report = {"kind": "synthetic-software-demonstration", "dataset": provenance,
              "trained_epochs": args.epochs, "runs": [str(path) for path in run_dirs],
              "comparison": str(out / "comparison/main_table.csv"), "browser": browser,
              "checkpoint_reloads": reloads, "new_sample_screening": screening,
              "http": http,
              "e2e_scope": "synthetic cached-feature input through trained models, "
                           "evaluation, unlabelled screening and review export",
              "verification": {"checkpoint_reloads": "passed", "screening": "passed",
                               "http": "passed" if args.http else "not_requested"},
              "not_verified": ["nuScenes data", "VLM labelling", "VLM feature extraction",
                               "real-data model quality"],
              "warning": "Fixture metrics do not establish real driving-scene performance."}
    (out / "demo.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[SYNTHETIC] Four trained CPU runs, table and browser previews written to {out}")
    print(f"[SYNTHETIC] Browser: {shlex.join(launch)}")


if __name__ == "__main__":
    main()
