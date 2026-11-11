# NetFlow datasets

Download the three exact versions below from the University of Queensland. On each UQ download page, select **Download file**. The publisher lists the datasets as open access with commercial-use restrictions. See the original dataset records for terms and citation requirements.

| Dataset | UQ download | Dataset DOI | CSV required by the pipeline |
| --- | --- | --- | --- |
| NF-ToN-IoT-v1 | [Download page](https://rdm.uq.edu.au/files/2ad93cd0-ef9c-11ed-827d-e762de186848) | [10.48610/2fa2ed6](https://doi.org/10.48610/2fa2ed6) | `NF-ToN-IoT.csv` |
| NF-BoT-IoT-v1 | [Download page](https://rdm.uq.edu.au/files/48fa1900-ef9c-11ed-827d-e762de186848) | [10.48610/62e6d80](https://doi.org/10.48610/62e6d80) | `NF-BoT-IoT.csv` |
| NF-ToN-IoT-v3 | [Download page](https://rdm.uq.edu.au/files/343e2e8c-6e6e-4a0c-813d-a46acea1b7f4) | [10.48610/44d7c5e](https://doi.org/10.48610/44d7c5e) | `NF-ToN-IoT-v3.csv` |

## Experimental sample

| Characteristic | NF-ToN-IoT-v1 | NF-BoT-IoT-v1 | NF-ToN-IoT-v3 |
| --- | ---: | ---: | ---: |
| Original flows | 1,379,274 | 600,100 | 27,520,260 |
| Selected flows | 357,868 | 189,445 | 350,000 |
| Selected benign flows | 60,000 | 13,859 | 213,562 |
| Selected attack flows | 297,868 | 175,586 | 136,438 |
| Attack share of selected flows | 83.23% | 92.68% | 38.98% |
| Classes, including benign | 10 | 5 | 10 |
| Original feature columns in the CSV | 12 | 12 | 53 |
| Features used as classifier inputs | 10 | 10 | 49 |

The standard profile samples up to 60,000 flows per class for each v1 dataset. It selects 350,000 v3 flows by reservoir sampling, then orders the temporal evaluation stream using flow-start timestamps. Classifier inputs exclude endpoint IPs and raw timestamps. Flow-start timestamps are retained separately for the temporal analysis. `Label` is the binary target and `Attack` identifies the family.

## Verify the input files

Keep each extracted UQ folder intact under the project's lowercase `data/` directory so the pipeline can find its CSV, feature schema, and original BagIt manifest. The following SHA-1 values identify the **CSV contents** used in the archived run and are listed in the publisher manifests and results workbook.

| CSV | SHA-1 |
| --- | --- |
| `NF-ToN-IoT.csv` | `ee8ee11c7e08bee0fb3d8d196d5c5fd2520a44d2` |
| `NF-BoT-IoT.csv` | `1e276a289f95521375248d91a4bdec06db56840e` |
| `NF-ToN-IoT-v3.csv` | `24854ca3072ab7c2ade9ebfb202345a666a94cfe` |

Dataset methods and provenance are described in [Sarhan et al., 2021](https://doi.org/10.1007/978-3-030-72802-1_9) for v1 and [Luay et al., 2025](https://arxiv.org/abs/2503.04404) for v3. The [UQ NetFlow collection](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) provides the feature definitions and class counts.
