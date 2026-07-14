# Changing kel: usual versus flip-flop PK

An interactive Streamlit teaching app comparing the effect of changing elimination rate in:

1. A usual one-compartment extravascular PK scenario, where absorption is faster than elimination.
2. A flip-flop scenario, where absorption takes days but intrinsic elimination takes hours.

## Run on Windows 11

1. Install Python 3.11 or newer from python.org and tick **Add Python to PATH** during installation.
2. Extract this folder.
3. Double-click `setup_and_run.bat`. It installs the required packages and starts the app.

For later use, double-click `run_app.bat`, or run:

```bat
python -m streamlit run app.py
```

The app opens in your default browser, usually at `http://localhost:8501`.

## What is adjustable

- Dose, bioavailability and apparent volume.
- A native/baseline elimination half-life (default 30 minutes) and a comparison elimination half-life, with the corresponding kel values displayed.
- Usual absorption half-life in hours (default 6 minutes, to remain faster than native elimination).
- Flip-flop absorption half-life in days (default 10 days).

## Outputs

- Linear concentration-time profiles.
- Semi-log profiles to demonstrate the terminal slope.
- Absorption input and elimination output rates.
- Tmax, Cmax, AUC and apparent terminal half-life.

## Model

The app uses a one-compartment model with first-order absorption and elimination:

C(t) = F Dose ka / [V(ka - kel)] x [exp(-kel t) - exp(-ka t)]

For educational use only. It does not include distribution kinetics, nonlinear PK, multiple absorption processes, lag time, enterohepatic recycling or repeated dosing.
