# Virtual Color Mixing Lab (Simulation)

This is a pure software simulation of a color mixing experiment.

## Setup

```bash
python -m venv .venv
# Windows PowerShell: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Run backend

```bash
.\.venv\Scripts\python flask_backend.py
```

## Run UI

Open `color_mixing_lab_frontend.html` in your browser.

## Notes
- Mixture swatch is computed from R/Y/B volumes.
- Sensor swatch is computed from 8 channels using `colour-science` with sum-normalisation.
