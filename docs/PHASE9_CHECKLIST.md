# Phase 9 manual checklist

Check W10 of the Phase 9 scorecard (`writeup/phase9_scorecard.json`) passes only when every scored line below is ticked
(`- [x]`) AND `python -m writeup.verify_phase9 --network` confirms it through the public GitHub API or the live app.

## Scored (W10 Release, W8 Live app)

- [x] **Redeploy** the Streamlit app from `main`, keeping the URL https://predictive-maintenance-oee.streamlit.app (verified by W8: the live Home page title equals `app/views/home.py`'s). Done 2026-11-14: new app from `main` (Python 3.14), old `phase8` app deleted, subdomain moved; `writeup/phase9_evidence.json` live_app ok, About checksum equals the manifest.
- [x] **Tag and release** `v1.0.0` on a commit on `main`, with the body of `docs/RELEASE_NOTES.md` (verified: tag, release body, CI success on the tagged commit). Done 2026-11-14: `v1.0.0` on merge commit 16158a4; evidence tag_on_main, ci_green_on_tag and release_notes all true.
- [x] **Repository settings**: description, website and topics exactly as in `writeup/phase9_spec.json` `release` (verified through the public API). Done 2026-11-14: evidence description, homepage and topics (15) all true; two earlier topics outside the spec were removed.
- [x] **Social preview** image `docs/figures/phase9/social_preview.png` uploaded in Settings > General > Social preview (the public API cannot read it; verified by opening the repository link preview). Done 2026-11-14: the repository page's og:image is an uploaded image (repository-images.githubusercontent.com), pixel-identical to the committed file (1280 x 640, mean difference 0).

## Not scored (career actions, done by the author)

- Pin the repository on the GitHub profile.
- Add the project to the LinkedIn Featured section.
