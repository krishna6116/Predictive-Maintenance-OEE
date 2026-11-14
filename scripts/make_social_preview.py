"""Phase 9: the repository's social preview card (1280 x 640), numbers from writeup.facts.

Run:  python scripts/make_social_preview.py     -> docs/figures/phase9/social_preview.png
Upload it in GitHub: Settings > General > Social preview (the API cannot set it). Fonts differ by operating system,
so the image is made once and committed rather than re-rendered byte-identically in CI.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from writeup.facts import facts  # noqa: E402

BLUE, ORANGE, INK, GREY, BG = "#0072B2", "#E69F00", "#1A1A1A", "#555555", "#FFFFFF"


def font(size: int, bold: bool = False):
    names = (["segoeuib.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "DejaVuSans.ttf", "arial.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def main() -> None:
    F = {k: f.value for k, f in facts().items()}
    im = Image.new("RGB", (1280, 640), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1280, 14], fill=BLUE)
    d.text((70, 60), "Predictive Maintenance & OEE", font=font(64, True), fill=INK)
    d.text((70, 150), "Forecast each engine's remaining life from sensor data, schedule crews before it fails,",
           font=font(28), fill=GREY)
    d.text((70, 188), "and measure the effect on a simulated plant. Every number re-checked by code in CI.",
           font=font(28), fill=GREY)
    tiles = [
        (f"{F['rmse_lstm_fd003']:.2f}", f"{F['published_model']} forecast error (RMSE, cycles)",
         f"vs {F['rmse_xgb_fd003']:.2f} XGBoost, unseen NASA engines"),
        (f"{100 * F['oee_pred']:.1f}%", "Simulated OEE, predictive",
         f"vs {100 * F['oee_fixed']:.1f}% fixed interval, {100 * F['oee_rtf']:.1f}% run to failure"),
        (f"{F['failures_pred']}", "Simulated unplanned failures",
         f"vs {F['failures_fixed']} fixed interval, {F['failures_rtf']} run to failure"),
    ]
    for i, (big, label, sub) in enumerate(tiles):
        x = 70 + i * 390
        d.rectangle([x, 270, x + 360, 470], outline="#DDDDDD", width=2)
        d.text((x + 24, 290), big, font=font(72, True), fill=BLUE if i == 0 else ORANGE)
        d.text((x + 24, 385), label, font=font(20, True), fill=INK)
        d.text((x + 24, 420), sub, font=font(16), fill=GREY)
    d.text((70, 530), "Python · PostgreSQL · PyTorch · XGBoost · OR-Tools · Prefect · Streamlit · Power BI",
           font=font(24), fill=INK)
    d.text((70, 575), "Krishnakumar Balasundaram  ·  predictive-maintenance-oee.streamlit.app", font=font(22), fill=GREY)
    out = ROOT / "docs" / "figures" / "phase9" / "social_preview.png"
    im.save(out, optimize=True)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
