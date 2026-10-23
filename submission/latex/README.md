# ACM article template (acmart)

A starter for ACM journal and conference papers, using the official
`acmart` document class.

## Files

- `main.tex` — the paper. Start here.
- `references.bib` — bibliography, cited from `main.tex`.

## The class is deliberately NOT included

`acmart.cls` ships with every modern TeX distribution, so this download does
not bundle a copy. Shipping one would pin you to whatever version we packaged
rather than the version your own system already has — and ACM updates the
class when venue requirements change.

If your distribution does not have it: https://ctan.org/pkg/acmart

## Building

    pdflatex main
    bibtex   main
    pdflatex main
    pdflatex main

Three LaTeX passes: one to collect labels and citations, `bibtex` to build the
bibliography, then two more to resolve every reference. Fewer passes leave
`??` in the output.

## Choosing your format

The `\documentclass` option selects the layout your venue requires:

| Option       | Use for                                     |
|--------------|---------------------------------------------|
| `sigconf`    | most ACM conference proceedings             |
| `sigplan`    | SIGPLAN conferences                         |
| `acmsmall`   | ACM small-trim journals                     |
| `acmlarge`   | ACM large-trim journals                     |
| `manuscript` | submission/review copy, single column       |

Add `review` for line numbers, `anonymous` for double-blind submission —
e.g. `\documentclass[sigconf,review,anonymous]{acmart}`.

## Before you submit

Replace the `\acmConference` and `\acmBooktitle` placeholders with the exact
strings your venue gives you, and check the current author guide — ACM changes
requirements between years, and the class version your venue expects may be
newer than the one installed locally.

## Licence

`acmart` is licensed **LPPL-1.3**, © Association for Computing Machinery.
The class is not redistributed here; see https://ctan.org/pkg/acmart.

This starter document is a minimal example written for latextemplates.com and
carries no additional restrictions.
