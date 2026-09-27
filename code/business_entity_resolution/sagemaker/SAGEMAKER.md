# Running the pipeline on Amazon SageMaker

The pipeline is CPU- and RAM-bound (polars, rapidfuzz, LightGBM). **Use a large CPU instance, not a GPU**: a GPU would sit idle.

| instance | vCPU | RAM | expected full run (train + predict) |
|---|---|---|---|
| `ml.m5.12xlarge` (recommended) | 48 | 192 GiB | ~1–1.5 h |
| `ml.c5.18xlarge` | 72 | 144 GiB | ~1 h |
| `ml.m5.24xlarge` | 96 | 384 GiB | < 1 h |
| laptop, for comparison | 12 | 24 GiB | 3–3.5 h |

With 144+ GiB of RAM, the run fits the models on **100% of training entities** (`SAMPLE=1.0`), instead of the 30% the laptop can hold. Billing is per hour while the instance is running, so **stop it when you're done**.

## 1. Create the instance (AWS console)
1. Go to **Amazon SageMaker AI** → **Notebooks** → **Notebook instances** → **Create notebook instance**.
2. Fill in:
   - Name: `mesh-minds-er`
   - Instance type: **`ml.m5.12xlarge`**
   - Additional configuration → **Volume size: 100 GB**
3. Use the default IAM role, then **Create**. Wait until the status is *InService*, then choose **Open JupyterLab**.

In SageMaker Studio instead, create a JupyterLab space with the same instance type and 100 GB of storage.

## 2. Get code and data onto it
Open a **Terminal** in JupyterLab:
```bash
cd ~/SageMaker          # persistent disk (survives stop/start)
```
Choose one of these:
- **Upload the two zips** (made on the laptop) through the JupyterLab file browser (upload arrow), then:
  ```bash
  unzip -q mesh_minds_code.zip -d mesh-minds && cd mesh-minds && unzip -q ../dataset.zip
  ```
- **Or `git clone` the team repository**, if it contains the dataset. Then `cd` into it.

You should now have `dataset/train`, `dataset/test`, `code/` and `utils/` in the current folder.

## 3. Set up once, then run
```bash
bash code/business_entity_resolution/sagemaker/setup.sh
bash code/business_entity_resolution/sagemaker/run.sh
tail -f work_sm_full/run.log      # Ctrl-C only stops watching; the run keeps going
```
The run is detached (`nohup`), so closing the browser tab doesn't stop it. The log shows, in order:
- blocking recall
- recall after expansion
- stage-1 and stage-2 out-of-fold F0.5, by country
- the test predictions, ending with `self-check PASS`

## 4. Validate and download the submission
```bash
.venv/bin/python utils/validate_submission.py --matching output_sm_full/matching_results.tsv \
    --candidate output_sm_full/candidate_pairs.tsv --test-dir dataset/test --check-ids
```
Right-click `output_sm_full/matching_results.tsv` → **Download**, then upload it to the leaderboard. `work_sm_full/experiments.jsonl` records the run's settings and scores for the methodology document.

## 5. Stop the instance
Notebook instances → select → **Actions → Stop**. Everything in `~/SageMaker` is kept for next time.
