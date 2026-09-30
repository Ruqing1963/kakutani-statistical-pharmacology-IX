# Paper IX

`paper09_ad_gamma_secretase_gsm.tex` / `.pdf` — 17 pages, `amsart` 11pt.

Build with three passes of `pdflatex` from this directory:

```bash
pdflatex paper09_ad_gamma_secretase_gsm.tex   # three times
```

The build is clean: zero errors, zero warnings, zero overfull or underfull boxes.

The figure is read from `../figures/` via `\graphicspath`. Every number in the tables comes from
`../code/paper09_compute_tables.py`, whose output is in `../results/paper09_table*_rows.tex` and
`../results/paper09_tables_summary.txt`.
