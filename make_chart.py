"""Before/after chart: real M0 computed from the raw CPI vs the rebased CPI."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cashless import pipeline as p  # noqa: E402

raw = p.fix_m0(p.load_raw("data/raw/kenya_monthly_raw.xlsx"))
naive = raw["m0_nominal"] / (raw["cpi_index"].ffill() / 100)
clean = p.build("data/raw/kenya_monthly_raw.xlsx")

fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
ax.plot(raw["date"], naive, color="#c0392b", lw=1.4, label="Raw CPI (two base years mixed)")
ax.plot(clean["date"], clean["real_m0"], color="#1f4e79", lw=2, label="Pipeline output (CPI rebased)")
ax.axvline(p.COVID_START, color="#888", ls="--", lw=1)
ax.text(p.COVID_START, ax.get_ylim()[1] * 0.97, "  COVID dummy starts", color="#555", va="top", fontsize=9)
ax.set_title("Real currency in circulation, Kenya (KES bn)", loc="left", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig("docs/real_m0_before_after.png")
print("saved")
