r"""Structural validation of the LaTeX sources without a TeX toolchain.

Checks: every \input target exists, every \ref has a \label, every \cite key is in the .bib,
every \includegraphics file exists, environments balance, and no number in the manuscript
contradicts the generated tables.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
problems, notes = [], []


def read(p):
    return p.read_text() if p.exists() else ""


main = read(P / "main.tex")
appx = read(P / "supplementary" / "appendix.tex")
supp = "\n".join(read(P / "supplementary" / f) for f in
                 ("experimental_details.tex", "additional_results.tex"))
bib = read(P / "references.bib")
combined = main + "\n" + supp

# ---- \input targets ------------------------------------------------------
for src, base in ((main, P), (appx, P / "supplementary"), (supp, P)):
    for t in re.findall(r"\\input\{([^}]+)\}", src):
        # \tabledir is defined per-document (tables/ from main, ../tables/ from the
        # supplement); resolve it to the canonical location for checking
        t = t.replace("\\tabledir", "tables")
        cands = [base / t, base / (t + ".tex"), P / t, P / (t + ".tex")]
        if not any(c.exists() for c in cands):
            problems.append(f"missing \\input target: {t}")

# ---- figures -------------------------------------------------------------
for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", combined):
    if not (P / f).exists():
        problems.append(f"missing figure: {f}")

# ---- labels / refs -------------------------------------------------------
labels = set(re.findall(r"\\label\{([^}]+)\}", combined))
for r in set(re.findall(r"\\ref\{([^}]+)\}", combined)):
    if r not in labels:
        problems.append(f"\\ref to undefined label: {r}")
# equation refs use \eqref
for r in set(re.findall(r"\\eqref\{([^}]+)\}", combined)):
    if r not in labels:
        problems.append(f"\\eqref to undefined label: {r}")

# ---- citations -----------------------------------------------------------
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for m in re.findall(r"\\cite[a-z]*\{([^}]+)\}", combined):
    cited |= {k.strip() for k in m.split(",")}
for c in sorted(cited - keys):
    problems.append(f"citation not in references.bib: {c}")
for k in sorted(keys - cited):
    notes.append(f"reference defined but never cited: {k}")

# ---- environment balance -------------------------------------------------
for env in ("tabular", "table", "table*", "figure", "figure*", "equation", "align",
            "itemize", "enumerate", "center", "abstract"):
    o = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", combined))
    c = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", combined))
    if o != c:
        problems.append(f"unbalanced environment {env}: {o} begin, {c} end")

# ---- numbers in prose vs generated tables --------------------------------
tabs = "\n".join(read(p) for p in (P / "tables").glob("*.tex"))
must_appear = {
    "6.416": "primary monitor MAE", "0.087": "primary monitor MAE sd",
    "7.456": "ceiling MAE", "8.065": "strict a-priori MAE",
    "8.384": "online truncation MAE", "8.628": "privileged MAE",
    "21.01": "H3 predicted on B", "12.82": "H3 predicted on C", "19.97": "H3 predicted on D",
}
for v, what in must_appear.items():
    if v not in tabs:
        problems.append(f"value {v} ({what}) missing from generated tables")
    if v not in main and v not in supp:
        notes.append(f"value {v} ({what}) appears in tables but not in prose")

# ---- claims that must not reappear ---------------------------------------
banned = [
    (r"know(s)? when (it|they) (is|are) wrong", "anthropomorphic claim"),
    (r"training diversity (solves|fixes)", "overclaim on diversity"),
    (r"\bepistemic[- ]vs[- ]aleatoric\b", "withdrawn taxonomy"),
    (r"\bproves\b", "overclaim verb"),
    (r"\bgroundbreaking\b|\brevolutionary\b|\bfirst ever\b", "marketing language"),
]
for pat, why in banned:
    if re.search(pat, combined, flags=re.I):
        problems.append(f"banned phrasing present ({why}): /{pat}/")

# ---- required disclosures ------------------------------------------------
for pat, why in [(r"single seed", "H7 single-seed disclosure"),
                 (r"weak test", "H4 weak-test disclosure"),
                 (r"21\\?%|21\\%|\$21\\%\$|21\%", "censoring disclosure"),
                 (r"withdraw", "taxonomy withdrawal"),
                 (r"not explained by sample count alone", "size-matched wording")]:
    if not re.search(pat, combined, flags=re.I):
        problems.append(f"missing required disclosure: {why}")

print("=" * 64)
print(f"{'PROBLEMS':<12}{len(problems)}")
for p_ in problems:
    print("  x", p_)
print(f"\n{'NOTES':<12}{len(notes)}")
for n in notes:
    print("  -", n)
print("=" * 64)
print(f"files: main.tex {len(main.splitlines())} lines, "
      f"supplement {len(supp.splitlines())} lines, "
      f"{len(list((P/'tables').glob('*.tex')))} tables, "
      f"{len(list((P/'figures').glob('*.pdf')))} figures, {len(keys)} references")
sys.exit(1 if problems else 0)
