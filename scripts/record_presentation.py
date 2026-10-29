#!/usr/bin/env python3
"""Record an authorised real-image walkthrough of the shipped cached explorer.

Uses original local ZIPs and saved outputs. Navigation is scripted and captions
are presentation annotations; no model inference, fitting or human annotation
is performed. Private footage must not be committed to the public repository.
Requires Playwright, Chromium and FFmpeg in addition to the CPU project lock.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from verify_new_scene_review import serve_html
from verify_offline_browser import extract_payload, validate_payload

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    'success': 'a9def8f915cf4bfe9c74b864384ec6c7',
    'fp': 'b4ff30109dd14c89b24789dc5713cf8c',
    'fn': 'c1eed31234b94e9f8e22fbf3428b0ac2',
}
CHAPTERS = [
    (0, 10, '01 / A traceable CPU pilot',
     'Yike leads this team-capstone modernization with AI assistance. '
     'The recorded viewer uses 77 single-reviewer references after AI-label disclosure.'),
    (10, 24, '02 / One timestamp, six real views',
     'A reference-positive example: cached SAE score 6.490, above the frozen '
     'validation threshold 6.075. This is saved-model playback, not live inference.'),
    (24, 36, '03 / Inspect an activation direction',
     'Unit 67 links to high-activation examples. Numeric units have no validated '
     'semantic names; these examples support inspection, not a causal explanation.'),
    (36, 43, '04 / False positive: MLP fixed rule',
     'This clear-road frame has a normal reference. MLP probability 0.898 exceeds '
     'its fixed 0.5 threshold: an illustrative false positive.'),
    (43, 50, '04 / Keep decision policies separate',
     'The per-frame table separates the fixed prediction from the '
     'validation-selected policy. Error labels here are relative to the submitted references.'),
    (50, 57, '05 / False negative: the same MLP rule',
     'A bus stop and marked crossing are visible. MLP probability 0.065 is below '
     'the same 0.5 threshold, missing this positive reference.'),
    (57, 64, '05 / Preserve errors for review',
     'Both mistakes use the same saved model and fixed rule. Cases were selected '
     'after evaluation for explanation, not for tuning.'),
    (64, 77, '06 / Compare methods and retain negative findings',
     'The old test has 15 timestamps in two scenes. SAE top-four finds four '
     'reference positives; reward and no-reward weights are identical.'),
    (77, 84, '07 / Export the next review queue',
     'Export 77 candidate rows with blank new-review fields. '
     'No new judgments are created in this recording.'),
    (84, 90, '08 / Scope and next decision',
     'Pilot limits: one reviewer after AI disclosure, one seed, '
     'and three normal training frames from one scene.'),
]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp(seconds, separator=','):
    milliseconds = int(round(seconds * 1000))
    whole, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}{separator}{milliseconds:03}'


def add_caption(page, heading, text):
    page.evaluate('''({heading,text}) => {
      let box=document.getElementById('recording-caption');
      if(!box){
        box=document.createElement('div');box.id='recording-caption';
        box.style.cssText='position:fixed;left:0;right:0;bottom:0;z-index:99999;'
          +'background:rgba(13,29,52,.97);color:white;padding:16px 30px 20px;'
          +'border-top:3px solid #6fc5bf;font-family:Arial,sans-serif;pointer-events:none';
        box.innerHTML='<div id="recording-chapter"></div><div id="recording-text"></div>';
        document.body.append(box);document.body.style.paddingBottom='130px';
      }
      document.getElementById('recording-chapter').textContent=heading;
      document.getElementById('recording-chapter').style.cssText=
        'font-size:15px;color:#9bded8;margin-bottom:8px;font-weight:700';
      document.getElementById('recording-text').textContent=text;
      document.getElementById('recording-text').style.cssText=
        'font-size:23px;line-height:1.35;max-width:1360px';
    }''', {'heading': heading, 'text': text})


def show_sample(page, token):
    page.evaluate('token => {location.hash="#/sample/"+token}', token)
    page.wait_for_function(
        'token=>document.querySelector("#main .sub")?.textContent.includes(token)', arg=token)
    page.locator('#main').evaluate('e=>e.scrollIntoView({block:"start"})')
    page.wait_for_function('''()=>{
      const imgs=[...document.querySelectorAll('#main .views img')];
      return imgs.length===6&&imgs.every(i=>i.complete&&i.naturalWidth===1600
        &&i.naturalHeight===900);
    }''')


def highlight_mlp(page):
    row = page.locator('#main .comparison-table tr').filter(has_text='mlp')
    require(row.count() == 1, 'Expected one actual MLP comparison row')
    row.evaluate('e=>{e.style.background="#fff3d6";e.style.outline="3px solid #b27624";'
                 'e.scrollIntoView({block:"center"})}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', type=Path, default=ROOT / 'demo/driving_pilot.html')
    parser.add_argument('--parts', type=Path, nargs=4, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--browser-executable', type=Path, default=Path('/usr/bin/chromium'))
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    require(not output.exists(), 'Preserve earlier recordings; use a fresh output directory')
    output.mkdir(parents=True)
    payload, _ = extract_payload(args.html.read_text())
    require(validate_payload(payload) == 'human_ai_exposed', 'Reviewed-reference viewer required')
    require(payload['summary']['frames'] == 77, 'Expected the recorded 77-reference pilot')
    for name in ('fp', 'fn'):
        rows = payload['frames'][CASES[name]]['comparisons']
        row = next(row for row in rows if row['method'] == 'mlp')
        require(row['fixed_error'] == name and row['prediction'] == int(row['score'] >= .5),
                'MLP saved fixed-rule case differs')
    for part in args.parts:
        require(part.is_file(), 'Missing original ZIP part')
    media_cache = output / 'playwright-media' / 'ffmpeg-1011'
    media_cache.mkdir(parents=True)
    ffmpeg = shutil.which('ffmpeg')
    require(ffmpeg and shutil.which('ffprobe'), 'Installed FFmpeg and ffprobe are required')
    (media_cache / 'ffmpeg-linux').symlink_to(ffmpeg)
    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(media_cache.parent)
    report = {'schema': 'real-driving-presentation-v1', 'status': 'running',
              'scripted_navigation': True, 'original_images': True,
              'new_inference': False, 'new_training': False, 'human_annotation_performed': False,
              'audio': 'none; English captions', 'private_footage': True,
              'html_sha256': digest(args.html), 'recorder_sha256': digest(__file__),
              'zip_inputs': [{'filename': p.name, 'bytes': p.stat().st_size,
                              'sha256': digest(p)} for p in args.parts],
              'label_scope': payload['label_scope'], 'cases': CASES, 'chapters': [],
              'page_errors': [], 'http_errors': []}
    try:
        from playwright.sync_api import sync_playwright

        with serve_html(args.html.resolve()) as url, sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=str(args.browser_executable), headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage'])
            context_args = {'viewport': {'width': 1440, 'height': 1000}, 'accept_downloads': True}
            if not args.preview_only:
                context_args.update({'record_video_dir': str(output / 'raw-video'),
                                     'record_video_size': {'width': 1440, 'height': 1000}})
            context = browser.new_context(**context_args)
            video_clock = time.perf_counter()
            page = context.new_page()
            page.on('pageerror', lambda e: report['page_errors'].append(str(e)))
            page.on('response', lambda r: report['http_errors'].append(
                {'url': r.url, 'status': r.status}) if r.status >= 400 else None)
            page.set_default_timeout(45000)
            page.goto(url, wait_until='load')
            page.locator('#offline-parts').set_input_files([str(p.resolve()) for p in args.parts])
            page.locator('#offline-load').click()
            page.wait_for_function(
                '()=>document.getElementById("offline-status").textContent.includes("校验完成")',
                timeout=180000)
            page.locator('.offline-results').evaluate('e=>e.scrollIntoView({block:"start"})')
            demo_clock = time.perf_counter()
            report['recording_trim_start_seconds'] = demo_clock - video_clock
            for index, (start, end, heading, text) in enumerate(CHAPTERS):
                if not args.preview_only:
                    page.wait_for_timeout(max(0, (demo_clock + start - time.perf_counter()) * 1000))
                if start == 10:
                    show_sample(page, CASES['success'])
                elif start == 24:
                    page.locator('#main a.u').filter(has_text='#67').click()
                    page.wait_for_function(
                        '()=>document.querySelector("#main h2")?.textContent.includes("Unit 67")')
                    page.locator('#main').evaluate('e=>e.scrollIntoView({block:"start"})')
                    page.wait_for_timeout(600)
                elif start in (36, 50):
                    show_sample(page, CASES['fp' if start == 36 else 'fn'])
                elif start in (43, 57):
                    highlight_mlp(page)
                elif start == 64:
                    page.locator('.offline-results').evaluate(
                        'e=>e.scrollIntoView({block:"center"})')
                elif start == 77:
                    with page.expect_download() as pending:
                        page.locator('#review-export').click()
                    download = pending.value
                    path = output / 'review-queue.csv'
                    download.save_as(path)
                    rows = list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))
                    blank_review = all(not r['reviewer_label'] and not r['review_notes']
                                       for r in rows)
                    require(len(rows) == 77 and blank_review,
                            'Export must preserve 77 blank new-review rows')
                    report['review_export'] = {'rows': len(rows), 'sha256': digest(path),
                                               'new_review_fields_blank': True}
                add_caption(page, heading, text)
                page.wait_for_timeout(150)
                page.screenshot(path=str(output / f'chapter-{index + 1:02}.png'))
                report['chapters'].append({'start': start, 'end': end, 'heading': heading,
                                           'text': text, 'actual_seconds':
                                               time.perf_counter() - demo_clock})
            if not args.preview_only:
                page.wait_for_timeout(max(0, (demo_clock + 92 - time.perf_counter()) * 1000))
            video = page.video
            context.close()
            if not args.preview_only:
                report['raw_video'] = str(video.path())
            browser.close()
        require(not report['page_errors'] and not report['http_errors'], 'Browser errors occurred')
        srt = '\n\n'.join(f'{i+1}\n{timestamp(start)} --> {timestamp(end)}\n{text}'
                          for i, (start, end, _, text) in enumerate(CHAPTERS)) + '\n'
        (output / 'captions.srt').write_text(srt)
        if not args.preview_only:
            target = output / 'driving-walkthrough-90s.mp4'
            command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-ss',
                       str(report['recording_trim_start_seconds']), '-i', report['raw_video'],
                       '-t', '90', '-r', '25', '-c:v', 'libx264', '-preset', 'veryfast',
                       '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an',
                       str(target)]
            subprocess.run(command, check=True)
            probe = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams',
                                    '-of', 'json', str(target)], capture_output=True,
                                   text=True, check=True)
            info = json.loads(probe.stdout)
            require(abs(float(info['format']['duration']) - 90) < .2, 'Video must be 90 seconds')
            require(info['streams'][0]['codec_name'] == 'h264', 'MP4 must use H.264')
            require(target.stat().st_size < 24 * 1024 * 1024, 'Keep the video below 24 MiB')
            report['video'] = {'filename': target.name, 'bytes': target.stat().st_size,
                               'sha256': digest(target), 'duration_seconds':
                                   float(info['format']['duration']), 'codec': 'h264',
                               'width': 1440, 'height': 1000}
        report['status'] = 'preview_passed' if args.preview_only else 'completed'
        print(json.dumps({'status': report['status'], 'chapters': len(report['chapters']),
                          'video': report.get('video'), 'review_export': report['review_export']}))
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        (output / 'recording.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
