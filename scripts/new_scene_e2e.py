#!/usr/bin/env python3
"""Run real, unlabelled six-camera images through the existing CPU SAE workflow.

Consumes a hash-addressed acquisition manifest, not fabricated nuScenes tables.
Reuses the pinned encoder, trained SAE, screening and review exporter. No fitting,
new ground truth, dataset split or accuracy evaluation is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
from PIL import Image
import torch

import cpu_image_encoder as encoder
from data_manifest import validate_tokens
from extraction_contract import sha256
from prepare_driving import CAMERAS
from screen import CPU_IDENTITY_FIELDS, validate_feature_compatibility

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def validate_acquisition(acquisition, data_root, download=False):
    require(acquisition.get('schema') == 'drivelm-author-demo-acquisition-v1',
            'A verified DriveLM author-demo acquisition manifest is required')
    frames = acquisition['frames']
    validate_tokens(frames, source='new unlabelled acquisition')
    require(frames and all(row.get('label') is None and row.get('split') is None
                           for row in frames), 'New intake must have no labels or split')
    images = {}
    commit = acquisition['source_commit']
    require(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit),
            'Pinned source commit is required')
    source_prefix = ('https://raw.githubusercontent.com/OpenDriveLab/DriveLM/'
                     + commit + '/challenge/llama_adapter_v2_multimodal7b/data/nuscenes/')
    for record in acquisition['images']:
        relative = Path(record['path'])
        require(not relative.is_absolute() and '..' not in relative.parts
                and len(relative.parts) == 3 and relative.parts[0] == 'samples'
                and relative.parts[1] in CAMERAS, 'Unsafe camera image path')
        require(record['path'] not in images, 'Duplicate camera image path')
        path = (data_root / relative).resolve()
        require(data_root in path.parents, 'Camera image escaped data root')
        if not path.is_file() and download:
            require(record['source_url'] == source_prefix + relative.as_posix(),
                    'Downloads must use the pinned author GitHub source')
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(record['source_url'], timeout=20) as response:
                data = response.read(record['bytes'] + 1)
            require(len(data) == record['bytes'], 'Downloaded image size differs')
            require(hashlib.sha256(data).hexdigest() == record['sha256'],
                    'Downloaded image hash differs')
            path.write_bytes(data)
        require(path.is_file() and path.stat().st_size == record['bytes']
                and sha256(path) == record['sha256'], 'Missing/changed image: ' + str(relative))
        with Image.open(path) as image:
            require(image.format == 'JPEG', 'Camera file is not JPEG')
            image.load()
            require([image.width, image.height] == record['width_height'],
                    'Decoded image dimensions differ')
        images[record['path']] = record
    meta, ordered_paths, referenced = [], [], []
    for row in sorted(frames, key=lambda item: (item['scene_token'], item['sample_token'])):
        require(set(row['image_paths']) == set(CAMERAS), 'Exactly six real views are required')
        paths = {}
        for camera in CAMERAS:
            source = row['image_paths'][camera]
            require(source.startswith('../nuscenes/samples/' + camera + '/'),
                    'Camera source path differs')
            relative = source.removeprefix('../nuscenes/')
            require(relative in images, 'Missing image manifest record')
            referenced.append(relative)
            paths[camera] = relative.removeprefix('samples/')
        timestamp = int(Path(paths['CAM_FRONT']).stem.rsplit('__', 1)[1])
        meta.append({'sample_token': row['sample_token'], 'scene_token': row['scene_token'],
                     'scene_name': row['scene_token'], 'timestamp': timestamp,
                     'timestamp_source': 'CAM_FRONT filename; not official sample timestamp',
                     'images': paths, 'row_index': len(meta)})
        ordered_paths.append([data_root / 'samples' / paths[camera] for camera in CAMERAS])
    require(len(referenced) == len(set(referenced)) == len(images),
            'Every image must belong to exactly one camera/frame')
    return meta, ordered_paths


def feature_contract(acquisition_path, acquisition, weights, meta, threads):
    revision = encoder.verify_weights(weights)
    return {'schema': encoder.SCHEMA, 'encoder': encoder.ENCODER,
            'model_revision': revision, 'model_files_sha256': {encoder.WEIGHTS_NAME: revision},
            'camera_order': list(CAMERAS), 'layers': encoder.LAYERS,
            'hook': encoder.HOOK, 'pooling': ['camera_concat'],
            'storage_aliases': {
                'layer1_mlp_output_mean.npy': 'camera_concat',
                'layer1_mlp_output_last_token.npy':
                    'camera_concat (identical compatibility alias)'},
            'dtype': 'float32', 'resize_short_side': encoder.RESIZE_SHORT_SIDE,
            'crop_size': encoder.CROP_SIZE, 'interpolation': 'PIL bilinear',
            'normalization_mean': [0.485, 0.456, 0.406],
            'normalization_std': [0.229, 0.224, 0.225],
            'view_feature_dim': encoder.VIEW_FEATURE_DIM, 'feature_dim': encoder.FEATURE_DIM,
            'input_source': 'DriveLM author-selected nuScenes demo; not full trainval',
            'source_commit': acquisition['source_commit'],
            'acquisition_sha256': sha256(acquisition_path),
            'samples': [{key: row[key] for key in ('sample_token', 'scene_token')}
                        for row in meta], 'device': 'cpu', 'threads': threads,
            'packages': {name: importlib.metadata.version(name)
                         for name in ('torch', 'numpy', 'Pillow')},
            'source_sha256': {name: sha256(Path(__file__).with_name(name))
                              for name in ('new_scene_e2e.py', 'cpu_image_encoder.py',
                                           'screen.py', 'sae_common.py', 'tail_activations.py',
                                           'neuron_explorer.py', 'export_unlabelled_review.py')},
            'labels_available': False, 'split_assigned': False,
            'original_log_metadata_available': False, 'log_disjoint_verified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('acquisition', 'data-root', 'weights', 'run-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='Fetch missing pinned author JPEGs')
    parser.add_argument('--training-meta', type=Path,
                        help='Relocated original cohort meta.json; recorded SHA256 must match')
    parser.add_argument('--threads', type=int, default=2, choices=(1, 2))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    data_root = args.data_root.resolve()
    output, run = args.output_dir.resolve(), args.run_dir.resolve()
    require(not output.exists(), 'Use a fresh output directory')
    acquisition = read_json(args.acquisition)
    meta, paths = validate_acquisition(acquisition, data_root, args.download)
    model_metadata = read_json(run / 'run_metadata.json')
    original_meta_path = args.training_meta or Path(model_metadata['inputs']['meta']['path'])
    require(sha256(original_meta_path) == model_metadata['inputs']['meta']['sha256'],
            'Original training-cohort metadata differs')
    original_meta = read_json(original_meta_path)
    for field in ('sample_token', 'scene_token'):
        require(not ({row[field] for row in meta} & {row[field] for row in original_meta}),
                'New intake overlaps original cohort: ' + field)
    checkpoint_hash = sha256(run / 'best_model.pth')
    require(checkpoint_hash == model_metadata['checkpoint']['sha256'], 'Model checkpoint changed')
    standardizer_hash = sha256(run / 'standardizer.npz')
    contract = feature_contract(args.acquisition, acquisition, args.weights, meta, args.threads)
    provenance = model_metadata['config']['feature_provenance']
    require(all(contract.get(key) == provenance.get(key) for key in CPU_IDENTITY_FIELDS),
            'New extraction protocol differs from the trained model')
    output.mkdir(parents=True)
    extract = output / 'extract'
    extract.mkdir()
    write_json(extract / 'meta.json', meta)
    write_json(extract / 'extraction_config.json', contract)
    torch.set_num_threads(args.threads)
    encoder_model = encoder.load_encoder(args.weights)
    features = []
    for row, camera_paths in zip(meta, paths):
        features.append(encoder.encode_views(encoder_model, camera_paths))
        print(f'[new-scenes] extracted {len(features)}/{len(meta)}', flush=True)
    features = np.stack(features)
    require(features.shape == (len(meta), encoder.FEATURE_DIM)
            and np.isfinite(features).all(), 'Invalid extracted features')
    for name in ('layer1_mlp_output_mean.npy', 'layer1_mlp_output_last_token.npy'):
        np.save(extract / name, features)
    compatibility = validate_feature_compatibility(run, extract / 'layer1_mlp_output_mean.npy')
    commands = []

    def execute(script, *arguments):
        command = [sys.executable, str(ROOT / 'scripts' / script), *map(str, arguments)]
        commands.append(command)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        (output / (script + '.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
        require(result.returncode == 0, script + ' failed; see output log')

    execute('screen.py', '--run-dir', run, '--features', extract / 'layer1_mlp_output_mean.npy',
            '--meta', extract / 'meta.json', '--output-dir', output / 'screening',
            '--device', 'cpu')
    execute('neuron_explorer.py', 'sample', meta[0]['sample_token'],
            '--run', output / 'screening', '--samples-root', data_root / 'samples',
            '--save', output / 'six-view-preview.jpg')
    for mode, filename in (('local', 'review.html'), ('source', 'review_source.html')):
        execute('export_unlabelled_review.py', '--run', output / 'screening',
                '--samples-root', data_root / 'samples', '--source-manifest', args.acquisition,
                '--image-mode', mode, '--output', output / filename)
    shutil.copytree(data_root / 'samples', output / 'images')
    require(sha256(run / 'best_model.pth') == checkpoint_hash
            and sha256(run / 'standardizer.npz') == standardizer_hash,
            'Inference must not mutate model weights or training normalisation')
    write_json(output / 'commands.json', commands)
    screened = read_json(output / 'screening' / 'screening_metadata.json')
    report = {'schema': 'new-scene-real-cpu-e2e-v1', 'status': 'completed',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'frames': len(meta), 'images': len(acquisition['images']),
              'scenes': len({row['scene_token'] for row in meta}),
              'feature_shape': list(features.shape), 'device': 'cpu',
              'feature_protocol_validation': compatibility, 'old_scene_overlap': 0,
              'old_sample_overlap': 0, 'log_disjoint_verified': False,
              'model_checkpoint_sha256': checkpoint_hash,
              'standardizer_sha256': standardizer_hash, 'training_assets_unchanged': True,
              'model_training_label_scope': model_metadata['label_scope'],
              'current_data_labels_available': False, 'training_performed': False,
              'accuracy_evaluation_performed': False,
              'flagged_by_fixed_model_rule': screened['flagged'],
              'decision': screened['decision'],
              'feature_sha256': sha256(extract / 'layer1_mlp_output_mean.npy'),
              'acquisition_sha256': sha256(args.acquisition),
              'review_html_sha256': sha256(output / 'review.html'),
              'review_source_html_sha256': sha256(output / 'review_source.html'),
              'elapsed_seconds': time.perf_counter() - started,
              'browser_ui_verified': False,
              'scope': 'Author-selected unlabelled new-scene inference/review smoke; '
                       'no accuracy claim'}
    write_json(output / 'flow.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
