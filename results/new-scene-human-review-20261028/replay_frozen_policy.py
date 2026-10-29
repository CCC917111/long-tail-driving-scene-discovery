#!/usr/bin/env python3
"""Replay saved SAE policies on the supplemental review, without selecting one.

Only recorded JSON/CSV inputs and Python's standard library are needed. Optional
private training directories add verification of the actual checkpoint bytes;
they are not necessary to reconstruct decisions from saved scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import csv


METHODS = ('sae_abstopk_none', 'sae_abstopk_norm')
ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return Path(path).read_text(encoding='utf-8')


def load_json(path):
    return json.loads(read_json(path))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def metrics(records, field):
    confusion = Counter({'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})
    selected = []
    for row in records:
        positive, prediction = row['reference_label'] == 1, row[field] == 1
        key = ('tp' if positive else 'fp') if prediction else ('fn' if positive else 'tn')
        confusion[key] += 1
        if prediction:
            selected.append(row['group'])
    tp, fp, fn, tn = (confusion[k] for k in ('tp', 'fp', 'fn', 'tn'))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {'confusion': dict(confusion), 'support': {'normal': tn + fp, 'tail': tp + fn},
            'selected': tp + fp, 'selected_groups': selected, 'tail_references_found': tp,
            'precision': precision, 'recall': recall,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'accuracy': (tp + tn) / len(records),
            'review_items_not_selected': len(records) - tp - fp,
            'review_item_reduction_fraction': (len(records) - tp - fp) / len(records),
            'false_positive_groups': [r['group'] for r in records
                                      if not r['reference_label'] and r[field]],
            'false_negative_groups': [r['group'] for r in records
                                      if r['reference_label'] and not r[field]]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-results', type=Path,
                        default=ROOT / 'results/human-ai-exposed-20261026')
    parser.add_argument('--new-inference', type=Path,
                        default=ROOT / 'results/new-scene-e2e-20261027')
    parser.add_argument('--review-results', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--private-training-root', type=Path,
                        help='Optional original comparison_runs directory; verify weight bytes')
    parser.add_argument('--output-prefix', type=Path,
                        default=Path(__file__).resolve().with_name('frozen_policy_evaluation'))
    args = parser.parse_args(argv)
    old, inference, review = (
        path.resolve() for path in (args.old_results, args.new_inference, args.review_results))
    public = load_json(review / 'frozen_policy_public_source.json')
    protocol = load_json(old / 'protocol_plan.json')
    flow = load_json(inference / 'flow.json')
    screening = load_json(inference / 'screening_metadata.json')
    references = load_json(review / 'normalized_references.json')
    provenance = load_json(review / 'provenance.json')
    audit = load_json(review / 'audit.json')
    require(screening['kind'] == 'unlabelled-screening'
            and screening['decision'] == public['fixed_rule'] == flow['decision'],
            'New inference fixed activation decision differs')
    require(flow['created_utc'] < audit['created_utc'],
            'Saved new-data inference did not predate the human review audit')
    original = review / 'reviewed-original.csv'
    require(digest(original) == audit['source_csv_sha256'], 'Original human CSV changed')
    with original.open(newline='', encoding='utf-8-sig') as handle:
        submitted = list(csv.DictReader(handle))
    require(provenance['human_reviewed'] is True
            and provenance['independent_human_review'] is False
            and provenance['ai_reference_disclosed_in_chat_before_submission'] is True,
            'Supplemental reference disclosure scope is required')
    with (inference / 'screening.csv').open(newline='', encoding='utf-8-sig') as handle:
        scores = list(csv.DictReader(handle))
    require(len(references) == len(scores) == 9, 'Expected the original nine-frame batch')
    require(len({r['sample_token'] for r in references}) == len(references)
            and len({r['sample_token'] for r in scores}) == len(scores), 'Duplicate input tokens')
    ref_of = {r['sample_token']: r for r in references}
    require(set(ref_of) == {r['sample_token'] for r in scores}, 'Scores/references differ')
    require(len(submitted) == len(references)
            and {r['sample_token'] for r in submitted} == set(ref_of),
            'Original and normalized review tokens differ')
    for group, row in enumerate(submitted, 1):
        reference = ref_of[row['sample_token']]
        require(row['review_status'] == 'reviewed' and row['reviewer'].strip()
                and row['label'] == reference['category']
                and reference['label'] == {'normal_core': 0, 'not_normal_core': 1}[row['label']]
                and reference['group'] == group
                and reference['review_notes'] == row['review_notes'],
                'Normalized review differs from original human submission')
    require(all(r['label'] in (0, 1) and r['split'] is None for r in references),
            'Expected unsplit binary reviewed references')
    require(protocol['review_k'] == 4, 'Original review budget differs')
    methods, files = {}, [original, review / 'provenance.json',
                          review / 'normalized_references.json',
                          review / 'frozen_policy_public_source.json',
                          inference / 'screening.csv', inference / 'screening_metadata.json',
                          inference / 'flow.json', old / 'protocol_plan.json']
    for method in METHODS:
        directory = old / method
        evaluation = load_json(directory / 'evaluation.json')
        metadata = load_json(directory / 'run_metadata.json')
        point = evaluation['operating_point']
        require(point == public['operating_point'],
                'Old validation point differs from public commit')
        require(evaluation['fixed_rule'] == public['fixed_rule'], 'Fixed activation rule differs')
        require(point['selection_split'] == 'val' and point['status'] == 'selected'
                and point['comparison'] == 'score >= threshold', 'Unexpected old policy')
        require(metadata['config']['review_k'] == protocol['review_k'], 'Budget changed')
        require(metadata['checkpoint']['sha256'] == flow['model_checkpoint_sha256']
                == screening['checkpoint_sha256'], 'Screened/trained model identities differ')
        require(metadata['created_utc'] < flow['created_utc'],
                'Model was not recorded before intake')
        require(public['commit_committed_at'] < flow['created_utc'],
                'The immutable published policy did not predate new inference')
        private_check = {'actual_checkpoint_bytes_verified': False,
                         'actual_standardizer_bytes_verified': False}
        if args.private_training_root:
            private = args.private_training_root / method
            require(digest(private / 'best_model.pth') == screening['checkpoint_sha256'],
                    'Actual checkpoint hash differs')
            require(digest(private / 'standardizer.npz') == screening['standardizer_sha256']
                    == flow['standardizer_sha256'], 'Actual training standardizer differs')
            private_check = {'actual_checkpoint_bytes_verified': True,
                             'actual_standardizer_bytes_verified': True}
        methods[method] = {'created_utc': metadata['created_utc'],
                           'checkpoint_sha256': metadata['checkpoint']['sha256'],
                           'operating_point': point, 'fixed_rule': evaluation['fixed_rule'],
                           'review_k': metadata['config']['review_k'], **private_check}
        files.extend((directory / 'evaluation.json', directory / 'run_metadata.json'))
    require(methods[METHODS[0]]['checkpoint_sha256'] == methods[METHODS[1]]['checkpoint_sha256'],
            'Expected identical recorded SAE variants')
    point = methods[METHODS[0]]['operating_point']
    threshold = float(point['threshold'])
    require(math.isfinite(threshold), 'Invalid saved threshold')
    ranked = sorted(scores, key=lambda r: (-float(r['tail_score']), r['sample_token']))
    require(all(math.isfinite(float(r['tail_score'])) for r in ranked), 'Invalid saved score')
    top = {r['sample_token'] for r in ranked[:protocol['review_k']]}
    rows = []
    for saved in scores:
        token = saved['sample_token']
        reference = ref_of[token]
        require(saved['scene'] == reference['scene_token'], 'Score/reference scenes differ')
        fixed = int(int(saved['active_tail_units']) >= 1)
        require(saved['prediction'] == ('long_tail' if fixed else 'normal'),
                'Saved fixed prediction differs from its activation count')
        rows.append({'group': reference['group'], 'sample_token': token,
                     'scene_token': reference['scene_token'], 'reference_label': reference['label'],
                     'reference_category': reference['category'],
                     'tail_score': float(saved['tail_score']),
                     'active_tail_units_at_saved_eta': int(saved['active_tail_units']),
                     'fixed_activation_prediction': fixed,
                     'frozen_validation_threshold': threshold,
                     'frozen_validation_prediction': int(float(saved['tail_score']) >= threshold),
                     'fixed_budget_k': protocol['review_k'],
                     'fixed_budget_prediction': int(token in top)})
    rows.sort(key=lambda r: r['group'])
    policies = {name: metrics(rows, field) for name, field in (
        ('fixed_activation_rule', 'fixed_activation_prediction'),
        ('frozen_validation_operating_point', 'frozen_validation_prediction'),
        ('predeclared_budget_k4', 'fixed_budget_prediction'))}
    report = {'schema': 'supplemental-frozen-policy-replay-v1',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'completed',
              'scope': provenance, 'frames': len(rows),
              'scenes': len({r['scene_token'] for r in rows}),
              'reference_counts': dict(Counter(r['reference_category'] for r in rows)),
              'old_public_policy': public, 'methods': methods,
              'sae_variants_identical': True,
              'duplicate_variants_are_not_independent_experiments': True,
              'policies': policies,
              'budget_random_expected_tail_references': (
                  protocol['review_k'] * sum(r['reference_label'] for r in rows) / len(rows)),
              'random_baseline_is_expectation_not_an_experiment': True,
              'threshold_selection_on_this_batch': False, 'training_performed': False,
              'new_split_assigned': False, 'policy_selected_based_on_new_results': False,
              'interpretation': (
                  'Descriptive replay on nine author-selected frames from two scenes. '
                  'Labels and scene are confounded; AI reference disclosure preceded submission. '
                  'The old validation precision target is not guaranteed on new data. '
                  'Review reduction counts items, not measured human time or generalization.'),
              'source_sha256': digest(__file__),
              'input_sha256': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
                               digest(p) for p in files}}
    output = args.output_prefix
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    output.with_suffix('.csv').write_text(stream.getvalue(), newline='')
    print(json.dumps({'status': report['status'], 'frames': len(rows), 'threshold': threshold,
                      'policies': policies}, indent=2))


if __name__ == '__main__':
    main()
