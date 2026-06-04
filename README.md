# OmniContact Project Page

This folder contains the offline project page for **OmniContact: Chaining Meta-Skills via Contact Flow for Generalizable Humanoid Loco-Manipulation**.

## Quick View

For a quick check, open:

```text
index.html
```

Most images and videos are stored locally and should be visible by directly double-clicking `index.html`.

## If Something Does Not Display

Some browsers restrict local `file://` access for interactive WebGL, iframe, JSON, CSV, WASM, or ONNX assets. If any section appears blank or fails to load, please serve this folder with a local HTTP server:

```text
python -m http.server 8765
```

Then open:

```text
http://localhost:8765/index.html
```

Run the command from this directory:

```text
omnicontact.github.io/
```

## Direct Video Access

If the webpage has display issues, the main video assets can be inspected directly under:

```text
static/videos/omnicontact/
```

Baseline videos are under:

```text
static/videos/omnicontact/baselines/
```

These files can be opened directly with a local video player.
