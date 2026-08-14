# Building the paper

```
paper/
├── main.tex                     # manuscript
├── references.bib               # 20 references
├── figures/                     # 8 vector PDFs, generated
├── tables/                      # 7 LaTeX tables, generated
└── supplementary/
    ├── appendix.tex             # standalone supplement (compiles on its own)
    ├── experimental_details.tex
    └── additional_results.tex
```

## Compile

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

```bash
cd paper/supplementary && pdflatex appendix && bibtex appendix && pdflatex appendix
```

`main.tex` compiles standalone as a two-column article. To use the official ICML style, drop
`icml2025.sty` (and its `.bst`) beside `main.tex`; the `\IfFileExists` switch at the top picks
it up automatically, so no edit is needed. The manuscript is always buildable without it.

**Not compiled here.** No TeX toolchain is installed on this machine, so the sources were
validated structurally instead — see below. Compile before submitting.

## Regenerate figures and tables

Both scripts read only stored artifacts under `results/`. Neither re-runs an experiment or
recomputes anything from features, so outputs cannot drift from the frozen results.

```bash
./.venv/bin/python scripts/make_paper_figures.py
```

```bash
./.venv/bin/python scripts/make_paper_tables.py
```

Every number in the manuscript that could be mistyped is emitted by `make_paper_tables.py`
rather than written by hand.

## Validate

```bash
./.venv/bin/python scripts/check_paper.py
```

Checks that every `\input` target and `\includegraphics` file exists, every `\ref`/`\eqref`
resolves to a `\label`, every `\cite` key is in `references.bib`, environments balance, key
numbers agree between prose and generated tables, required disclosures are present
(single-seed H7, weak-test H4, 21% censoring, taxonomy withdrawal, size-matched wording), and
that banned phrasings have not reappeared (anthropomorphic claims, the withdrawn
epistemic/aleatoric taxonomy, "proves", marketing language). Exit status is non-zero on any
problem. Current status: **0 problems**.

This is a substitute for, not a replacement of, a real compile.

## Before submission

1. **Compile** with a TeX toolchain and fix any errors this checker cannot see (spacing,
   overfull boxes, float placement, page limit).
2. **Verify the recent-preprint references.** `docs/phase0_literature.md` records a Phase-0
   survey that includes several 2026 preprints on certified prediction horizons, trust
   horizons, and self-estimated rollout error. Their bibliographic records were **not**
   verified from the arXiv listing within this repository, so they are deliberately **absent**
   from `references.bib`; `main.tex` points at the survey file instead of citing them. Verify
   them against arXiv and add proper entries, or drop the pointer.
3. **Check the page limit.** The manuscript currently runs long for a main track; §7
   (mechanistic analysis) and §5.7 (planning) compress most safely. Do not cut negative
   results — move them to the supplement with a forward reference if space demands it.
4. **Anonymise** if the venue requires it: the author block is empty, but the repository URL
   and Kaggle account names appear in the supplement's compute paragraph.

## Provenance

| Document | Purpose |
|---|---|
| `AUDIT.md` | Pre-rewrite audit: structure, claims, evidence, inconsistencies found |
| `CLAIM_AUDIT.md` | Every claim with its evidence, strength grade, and required wording |
| `PAPER.md` | Original draft, preserved unchanged |
| `PAPER_ICML.md` | Polished markdown mirror of the manuscript |
| `PREREGISTRATION*.md` | Five frozen protocols |
| `DEVIATIONS.md` | Ten logged departures with timing relative to test evaluation |
| `RESULTS.md` | Full results for all nine hypotheses |

Three numerical inconsistencies found in the original draft were fixed at the source (the
table generator) rather than by editing prose: the out-of-distribution table cited a different
experiment's arm, the headline row mixed statistics from two different computations, and two
minor values were misquoted. `AUDIT.md` §8 documents all three.
