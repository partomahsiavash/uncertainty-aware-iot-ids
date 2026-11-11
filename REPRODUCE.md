# Reproduce the experiments

The archived experiment ran on Windows with Python 3.11.9, PyTorch 2.10.0 CPU, and the `standard` profile. Run the commands from the folder containing `security_iot_uncertainty_pipeline.py`. For an offline review of the recorded outputs, open the [results workbook](Security_AI_Complete_Results.xlsx) or extract the [artifact bundle](iot-ids-research-artifacts.zip).

## Set up the environment

Use Python 3.11 and a virtual environment.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Linux**

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

The requirements pin versions recorded in the archived metadata and list the other dependencies by package name. XGBoost was available in the archived run. Its version was unrecorded. Use `.venv/bin/python` in place of `python` on Linux or `.\.venv\Scripts\python.exe` on Windows for the following commands.

## Run a synthetic smoke test

```text
python security_iot_uncertainty_pipeline.py --project-root . --device cpu --smoke-test
```

This uses generated inputs and writes `security_article_output_smoke_test/`. Its metrics are for checking the execution path.

## Run the standard experiment

1. Download and extract the three packages from [Dataset records](DATASETS.md). Keep the extracted folders, schema files, and manifests together inside the project's lowercase `data/` directory.
2. Run the archived profile with the project directory explicitly supplied.

```text
python security_iot_uncertainty_pipeline.py --project-root . --profile standard --device cpu
```

The pipeline writes a timestamped run under `security_article_output/`. The original run log spans about two hours. Runtime depends on the machine and software environment. The `quick` and `publication` profiles use different samples and settings from the archived run.

## Locate the output

| Path within a run | Contents |
| --- | --- |
| `excel/Security_AI_Complete_Results.xlsx` | Per-seed metrics, calibration, conformal coverage, open-set results, temporal experiments, statistical comparisons, and runtime |
| `figures/` | Ten PNG figures |
| `models/` | PyTorch encoder and classifier checkpoints |
| `experiment_configuration.json`, `run_metadata.json` | Run settings and reported environment |
| `execution.log`, `result_record_counts.json` | Run timeline and the number of recorded results |

The bundled workbook, configuration, metadata, and log use portable path references in place of absolute paths from the original workstation. Workbook measurements and checkpoint files are preserved.
