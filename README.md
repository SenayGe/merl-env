# MeRL (Medical RL) Environment

`merl-env` is a small Python library for building RL evnironments for clinical tasks. You can tweak the evaluation tasks, add agent-tools, and run RL training on your version of clinical tasks.

Currently, evaluation scenarios are based on MIMIC-IV dataset only.

## Quick Start

Use Python 3.10+.

```bash
python3.10 -m pip install -e ".[dev]"
python3.10 -m pytest
```

To try the library end to end right now with the bundled example cases:

```bash
python3.10 scripts/build_tasks.py \
  --source-fixtures examples/smoke_fixtures.json \
  --out-dir ./artifacts \
  --task all \
  --diagnosis-max-samples-per-label 100 \
  --diagnosis-sampling-seed 7 \
  --preview-limit 1
```

<!-- `--diagnosis-max-samples-per-label` is a per-label cap for the `diagnosis` task, and it is applied before train/val/test split assignment. Sampling is deterministic for a given `--diagnosis-sampling-seed`. -->

## Build From BigQuery

Install the optional BigQuery dependency:

```bash
python3.10 -m pip install -e ".[dev,bigquery]"
```

Authenticate with ADC, for example by setting `GOOGLE_APPLICATION_CREDENTIALS` or using gcloud ADC.

You can build a diagnosis dataset directly from the public PhysioNet MIMIC-IV tables:

```bash
python3.10 scripts/build_tasks.py \
  --source-bigquery \
  --gcp-project YOUR_GCP_PROJECT \
  --task diagnosis \
  --out-dir ./artifacts/diagnosis_bq \
  --diagnosis-max-samples-per-label 100 \
  --diagnosis-sampling-seed 7 \
  --preview-limit 1
```

Build a discharge dataset the same way:

```bash
python3.10 scripts/build_tasks.py \
  --source-bigquery \
  --gcp-project YOUR_GCP_PROJECT \
  --task discharge \
  --out-dir ./artifacts/discharge_bq \
  --preview-limit 1
```

Currently, you can only build dattaset for `diagnosis` and `discharge` tasks only. Will soon add more tasks.

## MIMIC Access

You do not need MIMIC access to run the smoke tests in this repo.

To build artifacts from real clinical data, you need:

- approved access to the MIMIC dataset
- access to the BigQuery where your MIMIC-derived tables live
- `google-cloud-bigquery` installed via the optional `bigquery` extra
- ADC configured for the account or service account that has access to those tables

Right now, the intended flow is to build local environments from your approved data source first, then run training or evaluation from those local artifacts.

## Repo Layout

- `src/merl_env/`: package code
- `scripts/`: local build and some smoke-test to try the library
<!-- - `tests/`: unit and smoke coverage -->
- `examples/smoke_fixtures.json`: tiny local fixture set for quick testing
- BigQuery mode uses the current public PhysioNet MIMIC table layout by default

## TODO

- [x] Support task prompts, parsers, tools, and environments
- [x] Add local developer scripts for artifact builds and smoke runs
- [x] Add a first BigQuery source adapter for `diagnosis` and `discharge`
- [ ] Add BigQuery support for `icd`
- [ ] LLM-judge verifiers
- [ ] Evaluation and training runner
- [ ] Auto-mode: describe task and spin-up env (autoresearch but for environments)
- [ ] Expand data sources
