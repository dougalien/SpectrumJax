# Spectrum Jax

A small app for presenting volcano transects. Upload a file of distance and elevation along a line across a
volcano, press a button, and get two pictures that can be pasted into a slide:

1. **The ideal shapes plot.** The perfect shapes (triangle, circle, square and the shapes between them) on a map
   of *b* (how bowed the sides are) against flat-top width. Each transect is a numbered diamond on the same map.
2. **The difference from the ideal.** The real outline against the best-fitting perfect shape, and the difference
   between them (more or less material than the perfect shape, as a percentage of the volcano's height).

Under each picture is a short explanation, and the page has a "What to say in class" box.

## Run it

```
py -m pip install -r requirements_jax.txt
py -m streamlit run spectrumJax.py
```

A browser tab opens. Tick **Also include the example transects** and press **Make my figures** to see what the
output looks like before using your own files.

To use it on another computer, copy `spectrumJax.py` and `requirements_jax.txt`. The `samples/jax_examples` folder
is only needed for the example transects (the tick box disappears without it).

## What to upload

Two columns of numbers, one row per point: **distance** along the line, then **elevation**. CSV, TXT, TSV, DAT or
Excel (.xlsx). A header row is optional; commas, semicolons, tabs or spaces can separate the columns, and decimal
commas are understood. If the header names the columns (distance / length / x, elevation / height / z) the order
does not matter. Choose the units of each column on the page: metres, kilometres, feet or miles for distance;
metres, feet or kilometres for elevation.

The line must cross the **whole volcano through the summit** and run out onto flat ground on **both sides**. A line
that starts at the summit, or stops partway down a flank, cannot be measured; the app says so.

You can upload several files at once. Each is a separate numbered diamond; several lines through the same volcano
show how much the answer depends on the direction.

## Copying the pictures

Under each picture:

* **Copy image** puts the picture on the clipboard, ready to paste into PowerPoint, Google Slides or Word. It needs
  a normal browser (Chrome, Edge, Safari, recent Firefox). If the browser refuses, the button says so.
* Right-click the picture and choose **Copy image**. This works in every browser.
* **Download PNG** saves it as a file.

The pictures are on a white background at high resolution, so they look the same on a slide.

## Where the numbers come from

The measuring code is a copy, unchanged, of the research app (`volcanoShape.py`): it finds the summit and the base
of the volcano, then measures *b*, the flat-top width and B/H the same way for the perfect shapes and for your
transects. `validation/validate_spectrum_jax.py` checks that Spectrum Jax still gives identical numbers to the
research app on all 60 real transects in `samples/real`, and that the pictures, the file reader and the app itself
work.

## What it cannot tell you

One transect is one slice through one volcano, at the resolution of the elevation data. It shows the shape along
that line, not why the volcano has that shape. The page lists the limits in "What this can and cannot tell you".

## Check it still works

```
py validation/validate_spectrum_jax.py
```

All checks should pass (tested on Python 3.12 with streamlit 1.56, numpy 2.4, matplotlib 3.10).
