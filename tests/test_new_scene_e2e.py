"""New-scene intake contract checks using tiny generated software fixtures."""

from copy import deepcopy
import hashlib
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from new_scene_e2e import validate_acquisition  # noqa: E402
from prepare_driving import CAMERAS  # noqa: E402


@pytest.fixture
def intake(tmp_path):
    data_root = tmp_path / 'data'
    records, image_paths = [], {}
    commit = '1' * 40
    source_prefix = ('https://raw.githubusercontent.com/OpenDriveLab/DriveLM/' + commit
                     + '/challenge/llama_adapter_v2_multimodal7b/data/nuscenes/')
    for index, camera in enumerate(CAMERAS):
        relative = f'samples/{camera}/synthetic__{camera}__{1000 + index}.jpg'
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (12, 8), (10 + index * 20, 40, 90)).save(path)
        data = path.read_bytes()
        records.append({'path': relative, 'sha256': hashlib.sha256(data).hexdigest(),
                        'bytes': len(data), 'width_height': [12, 8],
                        'source_url': source_prefix + relative})
        image_paths[camera] = '../nuscenes/' + relative
    acquisition = {'schema': 'drivelm-author-demo-acquisition-v1', 'source_commit': commit,
                   'images': records,
                   'frames': [{'sample_token': 'synthetic-frame-1',
                               'scene_token': 'synthetic-scene-1', 'label': None,
                               'split': None, 'image_paths': image_paths}]}
    return acquisition, data_root


def test_six_camera_order_and_no_invented_labels_or_split(intake):
    acquisition, data_root = intake
    metadata, ordered_paths = validate_acquisition(acquisition, data_root.resolve())
    assert len(metadata) == len(ordered_paths) == 1
    assert [path.parent.name for path in ordered_paths[0]] == list(CAMERAS)
    assert metadata[0]['row_index'] == 0
    assert metadata[0]['sample_token'] == 'synthetic-frame-1'
    assert metadata[0]['scene_token'] == 'synthetic-scene-1'
    assert metadata[0]['timestamp'] == 1000
    assert metadata[0]['timestamp_source'] == (
        'CAM_FRONT filename; not official sample timestamp')
    assert 'label' not in metadata[0] and 'split' not in metadata[0]
    assert all(path.is_file() for path in ordered_paths[0])


@pytest.mark.parametrize('change', ['hash', 'bytes', 'shape', 'missing_file'])
def test_changed_source_cannot_reach_feature_extraction(intake, change):
    acquisition, data_root = intake
    record = acquisition['images'][0]
    if change == 'hash':
        record['sha256'] = '0' * 64
    elif change == 'bytes':
        record['bytes'] += 1
    elif change == 'shape':
        record['width_height'] = [8, 12]
    else:
        (data_root / record['path']).unlink()
    with pytest.raises(ValueError, match='Missing/changed image|dimensions differ'):
        validate_acquisition(acquisition, data_root.resolve())


@pytest.mark.parametrize('field,value', [('label', 0), ('label', 1), ('split', 'test')])
def test_labelled_or_partitioned_data_rejected_by_unlabelled_route(intake, field, value):
    acquisition, data_root = intake
    acquisition['frames'][0][field] = value
    with pytest.raises(ValueError, match='no labels or split'):
        validate_acquisition(acquisition, data_root.resolve())


def test_missing_camera_is_not_padded_or_duplicated(intake):
    acquisition, data_root = intake
    acquisition['frames'][0]['image_paths'].pop('CAM_BACK_RIGHT')
    with pytest.raises(ValueError, match='Exactly six real views'):
        validate_acquisition(acquisition, data_root.resolve())


def test_duplicate_token_rejected(intake):
    acquisition, data_root = intake
    acquisition['frames'].append(deepcopy(acquisition['frames'][0]))
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        validate_acquisition(acquisition, data_root.resolve())


def test_source_file_path_traversal_rejected(intake):
    acquisition, data_root = intake
    acquisition['images'][0]['path'] = 'samples/CAM_FRONT/../../elsewhere.jpg'
    with pytest.raises(ValueError, match='Unsafe camera image path'):
        validate_acquisition(acquisition, data_root.resolve())


def test_missing_image_download_rejects_unrelated_destination_before_network(intake, monkeypatch):
    acquisition, data_root = intake
    record = acquisition['images'][0]
    (data_root / record['path']).unlink()
    record['source_url'] = 'https://example.invalid/not-author-data.jpg'
    monkeypatch.setattr('urllib.request.urlopen',
                        lambda *a, **kw: pytest.fail('Unvalidated URL reached network'))
    with pytest.raises(ValueError, match='pinned author GitHub source'):
        validate_acquisition(acquisition, data_root.resolve(), download=True)


def test_non_jpeg_payload_rejected_even_if_manifest_hash_matches(intake):
    acquisition, data_root = intake
    record = acquisition['images'][0]
    path = data_root / record['path']
    Image.new('RGB', (12, 8), 'red').save(path, format='PNG')
    record['bytes'] = path.stat().st_size
    record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='not JPEG'):
        validate_acquisition(acquisition, data_root.resolve())
