# Phase 9 manual checklist

Check W10 of the Phase 9 scorecard (`writeup/phase9_scorecard.json`) passes only when every scored line below is ticked
(`- [x]`) AND `python -m writeup.verify_phase9 --network` confirms it through the public GitHub API or the live app.

## Scored (W10 Release, W8 Live app)

- [ ] **Redeploy** the Streamlit app from `main`, keeping the URL https://predictive-maintenance-oee.streamlit.app (verified by W8: the live Home page title equals `app/views/home.py`'s).
- [ ] **Tag and release** `v1.0.0` on a commit on `main`, with the body of `docs/RELEASE_NOTES.md` (verified: tag, release body, CI success on the tagged commit).
- [ ] **Repository settings**: description, website and topics exactly as in `writeup/phase9_spec.json` `release` (verified through the public API).
- [ ] **Social preview** image `docs/figures/phase9/social_preview.png` uploaded in Settings > General > Social preview (the public API cannot read it; verified by opening the repository link preview).

## Not scored (career actions, done by the author)

- Pin the repository on the GitHub profile.
- Add the project to the LinkedIn Featured section.
