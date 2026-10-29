"""Independently replay actual JPEG encoding and model reload for the new-scene run."""
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

PROJECT = Path('/workspace/long-tail-driving-scene-discovery')
sys.path.insert(0, str(PROJECT / 'scripts'))
import cpu_image_encoder as encoder
from sae_common import LongTailGuidedSAE
from tail_activations import ActivationStore, CAMERAS


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=PROJECT / 'output/new-scene-e2e-20261027')
    parser.add_argument('--data-root', type=Path,
                        default=PROJECT / 'dataset/drivelm-author-demo-20261027')
    parser.add_argument('--run', type=Path,
                        default=PROJECT / ('output/real-cpu-mini-20261023/'
                                           'human-ai-exposed-20261026/comparison_runs/'
                                           'sae_abstopk_none'))
    args = parser.parse_args()
    output, data_root, run = args.output.resolve(), args.data_root.resolve(), args.run.resolve()
    flow = read_json(output / 'flow.json')
    acquisition = read_json(data_root / 'acquisition.json')
    model_record = read_json(run / 'run_metadata.json')
    contract = read_json(output / 'extract/extraction_config.json')
    metadata = read_json(output / 'extract/meta.json')
    original = read_json(model_record['inputs']['meta']['path'])
    assert len(metadata) == 9 and len(acquisition['images']) == 54
    assert not {row['sample_token'] for row in metadata} & {
        row['sample_token'] for row in original}
    assert not {row['scene_token'] for row in metadata} & {
        row['scene_token'] for row in original}
    assert len({row['scene_token'] for row in metadata}) == 2
    assert all('label' not in row and 'split' not in row for row in metadata)
    assert flow['current_data_labels_available'] is False
    assert flow['accuracy_evaluation_performed'] is False
    assert flow['training_performed'] is False
    assert flow['log_disjoint_verified'] is False
    assert digest(data_root / 'acquisition.json') == flow['acquisition_sha256']
    assert digest(output / 'review.html') == flow['review_html_sha256']
    assert digest(output / 'review_source.html') == flow['review_source_html_sha256']
    decoded_images = 0
    for image_record in acquisition['images']:
        path = data_root / image_record['path']
        assert digest(path) == image_record['sha256']
        assert path.stat().st_size == image_record['bytes']
        with Image.open(path) as image:
            assert image.format == 'JPEG'
            image.load()
            assert [image.width, image.height] == image_record['width_height']
        bundle_path = output / 'images' / image_record['path'].removeprefix('samples/')
        assert digest(path) == digest(bundle_path)
        decoded_images += 1
    features = np.load(output / 'extract/layer1_mlp_output_mean.npy')
    alias = np.load(output / 'extract/layer1_mlp_output_last_token.npy')
    assert features.shape == (9, 3072) and features.dtype == np.float32
    assert np.isfinite(features).all() and np.array_equal(features, alias)
    assert digest(output / 'extract/layer1_mlp_output_mean.npy') == flow['feature_sha256']
    for name, expected in contract['source_sha256'].items():
        assert digest(PROJECT / 'scripts' / name) == expected, name
    torch.set_num_threads(2)
    weights = PROJECT / 'models/resnet18/resnet18-f37072fd.pth'
    pretrained = encoder.load_encoder(weights)
    reencoded = []
    for row in metadata:
        assert set(row['images']) == set(CAMERAS)
        paths = [data_root / 'samples' / row['images'][camera] for camera in CAMERAS]
        reencoded.append(encoder.encode_views(pretrained, paths))
    reencoded = np.stack(reencoded)
    feature_max_diff = float(np.max(np.abs(features - reencoded)))
    np.testing.assert_allclose(features, reencoded, rtol=0, atol=1e-7)
    assert digest(run / 'best_model.pth') == model_record['checkpoint']['sha256']
    assert digest(run / 'best_model.pth') == flow['model_checkpoint_sha256']
    assert digest(run / 'standardizer.npz') == flow['standardizer_sha256']
    with np.load(run / 'standardizer.npz') as standardizer:
        mean, std = standardizer['mean'], standardizer['std']
    normalized = ((features - mean) / std).astype(np.float32)
    config = model_record['config']
    model = LongTailGuidedSAE(input_dim=features.shape[1],
                             hidden_dim=int(config['hidden_dim']),
                             tail_ratio=float(config['tail_ratio']),
                             dropout=float(config['dropout']), k=int(config['k']),
                             sparsity=config['sparsity'], reward=config['reward'])
    model.load_state_dict(torch.load(run / 'best_model.pth', map_location='cpu',
                                    weights_only=True), strict=True)
    model.eval()
    with torch.inference_mode():
        _, _, activations = model.encode(torch.from_numpy(normalized))
    reloaded = activations.numpy()
    stored = ActivationStore(output / 'screening')
    assert stored.labels is None and stored.splits is None
    assert [row['sample_token'] for row in stored.samples] == [
        row['sample_token'] for row in metadata]
    activation_max_diff = float(np.max(np.abs(stored.dense() - reloaded)))
    np.testing.assert_allclose(stored.dense(), reloaded, rtol=1e-6, atol=1e-6)
    eta = flow['decision']['eta']
    counts = (np.abs(reloaded) > eta).sum(axis=1)
    scores = np.linalg.norm(reloaded.astype(np.float64), axis=1)
    with (output / 'screening/screening.csv').open(newline='') as handle:
        predictions = list(csv.DictReader(handle))
    row_of = {row['sample_token']: i for i, row in enumerate(metadata)}
    assert len(predictions) == len(metadata)
    score_diffs = []
    for row in predictions:
        index = row_of[row['sample_token']]
        score_diffs.append(abs(float(row['tail_score']) - scores[index]))
        assert int(row['active_tail_units']) == counts[index]
        assert row['prediction'] == ('long_tail' if counts[index] else 'normal')
    assert max(score_diffs) < 1e-6
    assert sum(counts > 0) == flow['flagged_by_fixed_model_rule']
    report = {'schema': 'independent-new-scene-e2e-audit-v1', 'status': 'passed',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'real_jpegs_decoded_and_hashed': decoded_images,
              'frames': len(metadata), 'scenes': 2, 'feature_shape': list(features.shape),
              'old_sample_overlap': 0, 'old_scene_overlap': 0,
              'labels_and_split_absent': True, 'log_disjoint_verified': False,
              'fresh_raw_image_reencoding_max_abs_diff': feature_max_diff,
              'reloaded_sae_activation_max_abs_diff': activation_max_diff,
              'reloaded_tail_score_max_abs_diff': float(max(score_diffs)),
              'prediction_count': len(predictions), 'flagged_by_fixed_rule': int(sum(counts > 0)),
              'model_and_standardizer_hashes_match': True,
              'bundled_image_bytes_match_original': True,
              'review_html_hashes_match': True,
              'browser_ui_executed_by_this_audit': False,
              'accuracy_evaluation_performed': False,
              'audit_source_sha256': digest(__file__)}
    (output / 'independent_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
