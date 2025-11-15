# Rhythmic Segment Analysis Illustrations in TikZ

Author: Bas Cornelissen (documentation comments auto-generated).

This directory hosts the LaTeX packages for drawing 2D and 3D RSA simplex illustrations with TikZ. All inline documentation blocks were produced automatically, so tweak the sources and rerun the generator whenever you change behavior.

## Usage

Place the `.sty` files on your TeX path (or next to your document) and load only what you need:

```tex
\usepackage{rsa-2d}  % or rsa-3d, both pull in rsa-common
```

The commands exported by `rsa-2d` and `rsa-3d` compose with standard TikZ environments, so you can interleave them with your own drawing logic. See the inline comments in each package for argument details and defaults.

## Segments and patterns

You introduce points through `\defineSegment{<Name>}{<x>}{<y>(,<z>)}`. The macro creates:

- `<Name>` — the raw coordinate with the components you supplied.
- `<Name>p` — the normalized “pattern” coordinate projected onto the simplex (x+y=1 in 2D, x+y+z=1 in 3D).

Both coordinates are stored as macro registers, so you can access their components anywhere: `\X{A}`, `\Y{Ap}`, `\Z{Bp}`, etc. All high-level drawing helpers expect just the base name; their internal logic decides whether to target the raw point or its projection, e.g. `\showSegment{A}` renders the raw point while `\showPattern{A}` targets `Ap`.

### Layers

The 3D toolkit declares dedicated TikZ layers to keep depth cues clear:

- `background` — axes, ticks, and coordinate drop-lines.
- `simplex` — the simplex surface, grids, and labels that should float above axes but below point markers.
- `main` — TikZ’s default layer for your own drawing commands.
- `foreground` — emphasis overlays (highlighted markers, callouts) that must sit on top.

Helpers such as `\showAxes`, `\showSimplex`, and `\traceSegmentCoords` pick the appropriate layer automatically, but you can wrap your own paths in `\begin{pgfonlayer}{<name>}` blocks to match the same ordering.

## Package overview

- `rsa-common.sty` — shared helpers: sans-serif font setup, the `simplexColor` palette, formatting helpers such as `\round`, plus the reusable `\showSegment/\showPattern/\labelPattern` markers.
- `rsa-2d.sty` — 2D simplex utilities (`\defineSegment`, projection helper bundles, axis rendering, and distance rulers). Optional arguments mirror the 3D API for consistent styling hooks.
- `rsa-3d.sty` — 3D simplex utilities (`\showSimplex`, grid/tick helpers, projection traces, etc.) built atop `tikz-3dplot`, also reusing the shared helpers.
