"""Preserve the uploaded review and compare frozen outputs without retraining."""
import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    for name in ('review-csv', 'frame-order', 'ai-reference', 'screening-csv', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.review_csv.open(encoding='utf-8-sig', newline='')))
    frames = read_json(args.frame_order)
    ai_rows = read_json(args.ai_reference)['rows']
    predictions = list(csv.DictReader(args.screening_csv.open(newline='')))
    expected = [row['sample_token'] for row in frames]
    assert len(rows) == len(expected) == 9
    assert [row['sample_token'] for row in rows] == expected
    assert len(set(expected)) == len(expected)
    assert {row['sample_token'] for row in predictions} == set(expected)
    assert {row['sample_token'] for row in ai_rows} == set(expected)
    frame_of = {row['sample_token']: row for row in frames}
    ai_of = {row['sample_token']: row for row in ai_rows}
    prediction_of = {row['sample_token']: row for row in predictions}
    mapping = {'normal_core': 0, 'not_normal_core': 1}
    scope = {
        'label_protocol': 'manual', 'human_reviewed': True, 'reviewer_count': 1,
        'human_ground_truth_verified': False, 'independent_human_review': False,
        'official_benchmark': False,
        'ai_reference_disclosed_in_chat_before_submission': True,
        'individual_judgment_timing_verified': False,
        'review_context': 'single_reviewer_submission_after_ai_reference_disclosure',
        'interpretation': 'Submission followed AI-reference disclosure in chat. '
                          'Per-row decision timing is unknown; this is not verified blind review.',
        'browser_model_signal_exposure_is_separate': True,
    }
    comparison, normalized, per_scene = [], [], defaultdict(Counter)
    confusion = Counter({'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})
    for row in rows:
        token = row['sample_token']
        assert row['label'] in mapping and row['review_status'] == 'reviewed'
        assert row['reviewer'].strip()
        assert row['model_signal_exposed'] in {'true', 'false'}
        assert row['review_context'] == (
            'post_model_signal_exposure_single_reviewer'
            if row['model_signal_exposed'] == 'true' else 'model_hidden_single_reviewer')
        frame, reference, saved = frame_of[token], ai_of[token], prediction_of[token]
        assert saved['prediction'] in {'long_tail', 'normal'}
        assert math.isfinite(float(saved['tail_score']))
        label, predicted = mapping[row['label']], int(saved['prediction'] == 'long_tail')
        outcome = ('tp' if label else 'fp') if predicted else ('fn' if label else 'tn')
        confusion[outcome] += 1
        per_scene[frame['scene_token']][row['label']] += 1
        normalized.append({'sample_token': token, 'scene_token': frame['scene_token'],
                           'group': frame['group'], 'label': label, 'category': row['label'],
                           'reviewer': row['reviewer'], 'review_notes': row['review_notes'],
                           'review_status': row['review_status'], 'split': None,
                           'original_browser_review_context': row['review_context'],
                           'original_browser_model_signal_exposed':
                               row['model_signal_exposed'] == 'true'})
        comparison.append({'group': frame['group'], 'sample_token': token,
                           'human_label': row['label'], 'ai_visual_label': reference['label'],
                           'agrees_with_ai_reference': row['label'] == reference['label'],
                           'sae_fixed_prediction': saved['prediction'],
                           'sae_tail_score': float(saved['tail_score']),
                           'error_relative_to_submitted_reference': outcome,
                           'human_notes': row['review_notes']})
    tp, fp, fn = confusion['tp'], confusion['fp'], confusion['fn']
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    report = {
        'schema': 'new-scene-human-review-audit-v1', 'status': 'completed',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': scope,
        'source_csv_sha256': sha256(args.review_csv),
        'source_csv_bytes': args.review_csv.stat().st_size,
        'rows': len(rows), 'unique_tokens': len(set(expected)),
        'label_counts': dict(Counter(row['label'] for row in rows)),
        'reviewers': sorted({row['reviewer'] for row in rows}),
        'pending': 0, 'uncertain': 0,
        'nonempty_human_notes': sum(bool(row['review_notes'].strip()) for row in rows),
        'ai_reference_agreement': {
            'matches': sum(row['agrees_with_ai_reference'] for row in comparison),
            'total': len(rows), 'is_ai_accuracy_estimate': False},
        'sae_frozen_fixed_rule': {
            'rule': 'at least one tail activation abs(z) > 0.01',
            'confusion_relative_to_submitted_reference': dict(confusion),
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall),
            'flagged': tp + fp, 'candidate_fraction': (tp + fp) / len(rows),
            'review_items_reduced_in_this_batch': len(rows) - tp - fp},
        'scene_label_counts': {scene: dict(counts) for scene, counts in per_scene.items()},
        'scene_label_confound': all(len(counts) == 1 for counts in per_scene.values()),
        'training_performed': False, 'split_assigned': False,
        'original_csv_modified': False, 'human_notes_invented': False,
        'inputs': {key: {'sha256': sha256(value), 'filename': value.name}
                   for key, value in vars(args).items() if key != 'output_dir'},
        'audit_source_sha256': sha256(__file__),
    }
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    files = ('reviewed-original.csv', 'provenance.json', 'normalized_references.json',
             'comparison.csv', 'comparison.json', 'audit.json')
    assert all(not (out / name).exists() for name in files), 'Preserve previous imports'
    shutil.copyfile(args.review_csv, out / 'reviewed-original.csv')
    assert sha256(out / 'reviewed-original.csv') == sha256(args.review_csv)
    for name, data in [('provenance.json', scope), ('normalized_references.json', normalized),
                       ('comparison.json', comparison), ('audit.json', report)]:
        (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=list(comparison[0]))
    writer.writeheader()
    writer.writerows(comparison)
    (out / 'comparison.csv').write_text(text.getvalue(), encoding='utf-8-sig', newline='')
    print(json.dumps({key: report[key] for key in
                      ('status', 'rows', 'label_counts', 'nonempty_human_notes',
                       'ai_reference_agreement', 'sae_frozen_fixed_rule', 'scene_label_counts')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
