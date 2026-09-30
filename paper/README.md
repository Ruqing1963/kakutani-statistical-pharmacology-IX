# Paper IX

`paper09_ad_gamma_secretase_gsm.tex` / `.pdf` — 23 pages, `amsart` 11pt.

Build with three passes of `pdflatex` from this directory:

```bash
pdflatex paper09_ad_gamma_secretase_gsm.tex   # three times
```

The build is clean: zero errors, zero warnings, zero overfull or underfull boxes.

Float placement: Tables 1–8 and Figures 1–2 all sit inside Section 5 (pp. 17–20), and a `\clearpage`
after the last float keeps Section 6 continuous. Table 9 is Section 6's own assessment table.

The two figures are read from `../figures/` via `\graphicspath`. Every number in the tables comes from
`../code/paper09_compute_tables.py`, whose output is in `../results/paper09_table*_rows.tex` and
`../results/paper09_tables_summary.txt`.
