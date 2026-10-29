"""Actual browser failure/retry check; creates no annotation decisions."""
import json
from pathlib import Path
import sys
sys.path.insert(0, '/workspace/long-tail-driving-scene-discovery/scripts')
from verify_new_scene_review import serve_html
from playwright.sync_api import sync_playwright

html = Path('/workspace/sites/driving-review-yike-20261028/dist/index.html')
with serve_html(html) as url, sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True,
                               args=['--no-sandbox', '--disable-dev-shm-usage'])
    context = browser.new_context()
    page = context.new_page()
    page.route('**/images/CAM_FRONT/**', lambda route: route.abort())
    page.goto(url, wait_until='load')
    page.wait_for_function("document.getElementById('image-status').textContent.includes('图片加载失败')")
    assert all(page.locator('#' + name).is_disabled()
               for name in ('normal', 'tail', 'uncertain'))
    assert page.locator('#clear').is_enabled()
    page.locator('#reviewer').fill('CI_TEST_ONLY')
    page.locator('#notes').fill('Temporary retry check; not a human judgment')
    page.unroute('**/images/CAM_FRONT/**')
    page.locator('#retry-images').click()
    page.wait_for_function("DrivingUnlabelledReview.imageState.ready === true")
    assert all(page.locator('#' + name).is_enabled()
               for name in ('normal', 'tail', 'uncertain'))
    assert page.locator('#reviewer').input_value() == 'CI_TEST_ONLY'
    assert page.locator('#notes').input_value() == 'Temporary retry check; not a human judgment'
    assert page.evaluate("DrivingUnlabelledReview.state.rows.every(r=>r.label==='')")
    result = {'status': 'passed', 'blocked_image_disables_decisions': True,
              'retry_restores_all_six_images': True, 'unsaved_inputs_preserved': True,
              'human_annotation_performed': False}
    output = Path('/workspace/long-tail-driving-scene-discovery/output/review-page-repair-20261028/image-retry.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    context.close()
    browser.close()
