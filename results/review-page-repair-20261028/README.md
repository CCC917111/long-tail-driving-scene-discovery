# Review-page image delivery repair

The owner-private review Site is published at:
https://yike-driving-review-20261028.upbeatperch12.chatgpt.site

It requires the owner to sign in. Its 54 original JPEGs are hosted on the same
private origin, so the browser can load them without contacting the upstream raw-GitHub
image URLs. No image pixels or model weights are included in this public evidence
folder. The Site audience has not been made public.

Review buttons remain disabled until all six images for the current frame have loaded.
Late events from a previous frame are ignored. Retrying images preserves typed
reviewer/notes; clearing a judgment remains available.

| Record | Meaning |
|---|---|
| [deployment.json](deployment.json) | Native publication success, actual Site URL and saved version |
| [browser_verification.json](browser_verification.json) | Real Chromium desktop/mobile checks using the same 54 original image assets as the deployed archive |
| [image_retry.json](image_retry.json) | An intentionally failed image disables decisions; retry restores six views without losing typed inputs |
| [verify_image_retry.py](verify_image_retry.py) | Exact failure/retry audit source; retains the original worker paths |
| [delivery_versions.json](delivery_versions.json) | Exact hosted HTML hash and separate newer public-fallback HTML/exporter hashes |

The browser checks ran against the prepared source using temporary localhost HTTP,
before publishing that exact static archive. They do not verify an authenticated browser
visit after deployment. The native deployment record confirms publication separately.
Automated UI decisions use CI_TEST_ONLY and are not human labels.

The [public fallback HTML](../../demo/driving_new_scenes.html) has the same
image-loading protections and loads pinned author images over the Internet. Its separate
GitHub Actions workflow tests that fallback, not the owner-private Site. Browsers unable
to reach raw.githubusercontent.com should use the private Site when authorized. The
original 2026-10-27 output HTML and run JSON remain preserved, alongside their [recorded
source snapshot](../new-scene-e2e-20261027/execution_source.zip).
