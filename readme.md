# Exploratory Bearable Data Dashboard

A small Streamlit data project for exploring CSV exports from the [Bearable](https://bearable.app/) tracking app.

The dashboard turns event-style exports into daily features, adds a set of configurable heuristic flags, and visualizes relationships among mood, energy, sleep, symptoms, and logged nutrition.

This is an **exploratory personal-analytics tool**, not a medical prediction or diagnostic system. The flags are thresholds for finding days worth inspecting, not validated health alerts.

## What it does

- parses Bearable CSV exports into a daily time series;
- normalizes mood, energy, sleep, and symptom records;
- derives rolling and lagged features such as 7-day sleep variability;
- builds a combined score from a selected symptom group;
- adds configurable threshold flags for exploratory review;
- visualizes same-day and lagged relationships;
- shows a timeline of triggered flags;
- exports the processed daily dataset as CSV.

## Data flow

```text
Bearable CSV export
        |
        v
clean + normalize
        |
        v
daily feature table
  |     |      |
  |     |      +--> symptom aggregation
  |     +---------> rolling sleep statistics
  +---------------> mood / energy / nutrition features
        |
        v
heuristic flags + exploratory charts
        |
        +--> Streamlit dashboard
        +--> processed CSV download
```

## Input

The app expects a Bearable CSV export containing the fields used by the parser, including `date formatted`, `rating/amount`, `category`, and `detail`.

There are two ways to provide a file.

### Manual upload

This is the portable default. Run the app and upload a Bearable CSV from the sidebar.

### Optional auto-load directory

Set `BEARABLE_EXPORT_DIR` to a folder containing Bearable exports:

```powershell
$env:BEARABLE_EXPORT_DIR = "C:\path\to\Bearable_export"
streamlit run streamlit_app.py
```

When configured, the app looks for files named like:

```text
Bearable App - Data Export. Generated DD-MM-YYYY.csv
```

and auto-loads the newest matching export. If the directory is missing or no file matches, manual upload remains available.

No personal filesystem path is required by the repository.

## Features

### Daily feature engineering

The parser derives or aggregates:

- average mood;
- average energy;
- sleep duration and bedtime;
- sleep quality;
- symptom severity columns;
- total symptom score;
- 7-day mood average and delta;
- 7-day sleep-duration standard deviation;
- simple nutrition/logging features.

### Heuristic flags

Sidebar controls let you change thresholds for:

- low reported energy;
- short sleep two days earlier;
- high 7-day sleep variability;
- unusually high combined key-symptom score;
- low estimated logged calories relative to the dataset median;
- few logged food items;
- no evening food logged.

These are intentionally labelled **heuristics**. They help select observations for review; they do not demonstrate a causal relationship or prescribe an action.

### Visualizations

The dashboard includes:

- energy vs. mood regression view;
- short-sleep vs. mood two days later comparison;
- key-symptom score over time with a LOESS trend;
- an interactive flag timeline;
- the processed dataframe and CSV download.

## Methodological caveats

This repository grew out of exploratory analysis, so its constraints are part of the project:

- thresholds are hand-selected and configurable, not clinically validated;
- lagged comparisons are observational and should not be read as causal;
- the selected symptom group came from a limited historical window;
- nutrition values use a small hard-coded lookup table and are coarse estimates;
- missing or inconsistent logging can look like a behavioral signal;
- correlations discovered in one personal dataset may not generalize.

A stronger next iteration would separate reusable parsing/features from the Streamlit UI and add tests around Bearable-format changes.

## Run locally

Requirements: Python 3.10+.

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens locally and prompts for a CSV unless `BEARABLE_EXPORT_DIR` points to a matching export directory.

## Stack

- Streamlit
- pandas / NumPy
- Matplotlib + seaborn
- Altair
- SciPy

## Project status

This is a compact exploratory-analysis prototype, not a production health application.

Its main portfolio value is the data path: converting a messy event export into a daily analytical table, making assumptions visible as feature/threshold code, and presenting lagged and rolling patterns interactively without requiring a backend.
