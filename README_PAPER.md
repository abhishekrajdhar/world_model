# Building the paper

```
paper/
├── main.tex                     # manuscript
├── references.bib               # 29 references, all primary-source verified
├── figures/                     # 8 vector PDFs, generated
├── tables/                      # 7 LaTeX tables, generated
└── supplementary/
    ├── appendix.tex             # standalone supplement (compiles on its own)
    ├── experimental_details.tex
    └── additional_results.tex
```

## Compile

Built and verified with Tectonic 0.17.0 (`brew install tectonic`), which fetches the required
LaTeX packages on demand and runs BibTeX automatically:

```bash
cd paper && tectonic -X compile main.tex
```

```bash
cd paper/supplementary && tectonic -X compile appendix.tex
```

A standard TeX Live installation works equally well:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`main.tex` compiles standalone as a two-column article. To use the official ICML style, drop
`icml2025.sty` (and its `.bst`) beside `main.tex`; the `\IfFileExists` switch at the top picks
it up automatically, so no edit is needed.

**Build status:** both documents compile clean — **0 errors, 0 undefined references, 0
undefined citations, 0 overfull boxes** in `main.tex`. Output is **9 pages** (8 pages of body,
references beginning on page 9) plus a **4-page** supplement, which fits the ICML limit of 8
pages excluding references.

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

This runs in a second and catches things a compile does not (silent claim drift,
prose/table disagreement), so it is worth running alongside the build rather than instead of it.

## Before submission

1. **Compile** with a TeX toolchain and fix any errors this checker cannot see (spacing,
   overfull boxes, float placement, page limit).
2. **References are verified.** All nine recent preprints were checked against their arXiv
   abstract pages (title, full author list, identifier, submission date) and are cited
   properly; see `BIBLIOGRAPHY_AUDIT.md`. Nothing further is required here.
3. **Page limit is met** at 8 body pages, but with no slack. If the ICML style file changes
   the layout, §7 (mechanistic analysis) and §6.8 (planning) compress most safely. Do not cut
   negative results — move them to the supplement with a forward reference instead.
4. **Anonymise for double-blind review.** The manuscript now carries an author block
   (Abhishek Rajdhar Dubey, Independent Researcher). ICML review is double-blind, so this
   must be removed for submission and restored for camera-ready. The supplement's compute
   paragraph also names the Kaggle platform; check venue policy before submitting.

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
| `BIBLIOGRAPHY_AUDIT.md` | Primary-source verification of every reference; four claim corrections |

Three numerical inconsistencies found in the original draft were fixed at the source (the
table generator) rather than by editing prose: the out-of-distribution table cited a different
experiment's arm, the headline row mixed statistics from two different computations, and two
minor values were misquoted. `AUDIT.md` §8 documents all three.
