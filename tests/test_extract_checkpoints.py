"""Checkpoint fault injection on synthetic arrays; no real VLM inference."""

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import extract  # noqa: E402


@pytest.fixture
def cache(tmp_path):
    (tmp_path / 'extraction_config.json').write_text('{"fixture": true}\n')
    return tmp_path


def values(count, captions=False):
    vector = [np.array([i + 1., i + 2.], dtype=np.float32) for i in range(count)]
    rows = {(1, kind): vector for kind in ('mean', 'last')}
    meta = [{'sample_token': f's{i}', 'scene_token': f'scene{i}', 'row_index': i}
            for i in range(count)]
    text = [{'sample_token': row['sample_token'], 'caption': 'synthetic fixture'}
            for row in meta] if captions else []
    return rows, meta, text


def state(cache):
    return json.loads((cache / extract.CHECKPOINT_STATE).read_text())


def write(cache, count, captions=False):
    extract.save_checkpoint(cache, [1], *values(count, captions))
    return state(cache)


def authoritative(cache, name):
    return cache / state(cache)['files'][name]['path']


def update_recorded_hash(cache, name):
    current = state(cache)
    current['files'][name]['sha256'] = extract.sha256(authoritative(cache, name))
    (cache / extract.CHECKPOINT_STATE).write_text(json.dumps(current))


def test_interruption_before_pointer_publish_preserves_previous_generation(cache, monkeypatch):
    original = write(cache, 1)

    def fail_publish(*args):
        raise OSError('simulated interruption before pointer publication')

    with monkeypatch.context() as patched:
        patched.setattr(extract, '_publish_state', fail_publish)
        with pytest.raises(OSError, match='before pointer'):
            write(cache, 2)
    assert state(cache) == original
    _, meta, _ = extract.load_checkpoint(cache, [1])
    assert len(meta) == 1
    assert json.loads((cache / 'meta.json').read_text()) == meta
    np.testing.assert_array_equal(np.load(cache / 'layer1_mlp_output_mean.npy'), [[1., 2.]])


def test_interrupted_consumer_export_is_repaired_from_committed_generation(cache, monkeypatch):
    previous = write(cache, 1, captions=True)

    def interrupt_mirror(output_dir, current):
        name = 'layer1_mlp_output_mean.npy'
        shutil.copyfile(output_dir / current['files'][name]['path'], output_dir / name)
        raise OSError('simulated interruption during mirror export')

    with monkeypatch.context() as patched:
        patched.setattr(extract, '_mirror_checkpoint', interrupt_mirror)
        with pytest.raises(OSError, match='during mirror'):
            write(cache, 2, captions=True)
    committed = state(cache)
    assert committed['previous_generation'] == previous['generation']
    assert committed['row_count'] == 2
    assert len(json.loads((cache / 'meta.json').read_text())) == 1
    rows, meta, captions = extract.load_checkpoint(cache, [1])
    assert len(meta) == len(captions) == 2
    np.testing.assert_array_equal(rows[(1, 'mean')], [[1., 2.], [2., 3.]])
    for name, record in committed['files'].items():
        assert extract.sha256(cache / name) == record['sha256']


@pytest.mark.parametrize('corruption', ['swap_same_shape', 'nan_same_shape', 'changed_width'])
def test_authoritative_feature_hash_corruption_is_rejected(cache, corruption):
    write(cache, 2)
    path = authoritative(cache, 'layer1_mlp_output_mean.npy')
    array = np.load(path)
    if corruption == 'swap_same_shape':
        array = array[::-1]
    elif corruption == 'nan_same_shape':
        array[0, 0] = np.nan
    else:
        array = array[:, :1]
    np.save(path, array)
    with pytest.raises(ValueError, match='hash/file mismatch'):
        extract.load_checkpoint(cache, [1])


@pytest.mark.parametrize('corruption', ['nan', 'width', 'rank'])
def test_feature_structure_is_validated_besides_hashes(cache, corruption):
    write(cache, 2)
    name = 'layer1_mlp_output_mean.npy'
    path = authoritative(cache, name)
    array = np.load(path)
    if corruption == 'nan':
        array[0, 0] = np.nan
    elif corruption == 'width':
        array = array[:, :1]
    else:
        array = array.ravel()
    np.save(path, array)
    update_recorded_hash(cache, name)
    with pytest.raises(ValueError, match='invalid shape, dtype or nonfinite'):
        extract.load_checkpoint(cache, [1])


def test_duplicate_metadata_tokens_rejected_even_with_matching_hash(cache):
    write(cache, 2)
    path = authoritative(cache, 'meta.json')
    meta = json.loads(path.read_text())
    meta[1]['sample_token'] = meta[0]['sample_token']
    path.write_text(json.dumps(meta))
    update_recorded_hash(cache, 'meta.json')
    with pytest.raises(ValueError, match='duplicate sample tokens'):
        extract.load_checkpoint(cache, [1])


def test_root_mirrors_are_recovered_instead_of_trusted(cache):
    committed = write(cache, 2)
    np.save(cache / 'layer1_mlp_output_mean.npy', np.array([[np.nan]]))
    (cache / 'meta.json').unlink()
    extract.load_checkpoint(cache, [1])
    for name, record in committed['files'].items():
        assert extract.sha256(cache / name) == record['sha256']


def test_only_current_and_previous_generations_are_retained(cache):
    oldest = write(cache, 1)
    previous = write(cache, 2)
    current = write(cache, 3)
    retained = {f'.checkpoints/{path.name}' for path in (cache / '.checkpoints').iterdir()}
    assert retained == {current['generation'], previous['generation']}
    assert current['previous_generation'] == previous['generation']
    assert not (cache / oldest['generation']).exists()


def test_untracked_legacy_arrays_are_not_adopted(cache):
    np.save(cache / 'layer1_mlp_output_mean.npy', np.ones((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match='Untracked legacy checkpoint'):
        extract.load_checkpoint(cache, [1])
    with pytest.raises(ValueError, match='Untracked legacy checkpoint'):
        write(cache, 1)


def test_contract_hash_mismatch_rejected(cache):
    write(cache, 1)
    (cache / 'extraction_config.json').write_text('{"different": true}\n')
    with pytest.raises(ValueError, match='contract hash mismatch'):
        extract.load_checkpoint(cache, [1])


def test_complete_resume_repairs_mirrors_and_refreshes_stale_summary(cache, monkeypatch):
    committed = write(cache, 2)
    (cache / 'meta.json').unlink()
    (cache / 'extraction_summary.json').write_text(json.dumps({
        'status': 'failed_or_interrupted', 'selection_complete': False,
        'rows_in_cache': 1, 'forward_seconds_including_image_reads': 12.5}))
    args = SimpleNamespace(layers=[1], save_every=1, max_samples=0, output_dir=cache,
                           meta_dir=cache, samples_root=cache, sample_tokens=None,
                           resume=True, model_revision='a' * 40)
    monkeypatch.setattr(extract, 'parse_args', lambda: args)
    monkeypatch.setattr(extract, 'build_samples', lambda *args: values(2)[1])
    monkeypatch.setattr(extract, 'build_contract', lambda *args: {'fixture': True})
    monkeypatch.setattr(extract, 'freeze_contract', lambda *args: None)

    def no_inference(*args):
        pytest.fail('A complete verified cache must not load or run a VLM')

    monkeypatch.setattr(extract, 'load_vlm', no_inference)
    extract.main()
    summary = json.loads((cache / 'extraction_summary.json').read_text())
    assert summary['selection_complete'] and summary['rows_in_cache'] == 2
    assert summary['new_rows_this_invocation'] == 0 and summary['cache_verified_on_resume']
    assert summary['checkpoint_generation'] == committed['generation']
    assert summary['forward_seconds_including_image_reads'] == 12.5
    assert summary['feature_files']['layer1_mlp_output_mean.npy']['sha256'] == (
        committed['files']['layer1_mlp_output_mean.npy']['sha256'])
