# Early Warning for At-Risk Students

Traditional machine learning versus deep learning on the Open University Learning Analytics Dataset (OULAD).

Summative project for Introduction to Machine Learning.


| Deliverable | Link |
|-------------|------|
| Google Drive folder (raw dataset zip and saved models) |https://drive.google.com/drive/folders/1ODw-Nt7TtsUmvoxONtjImycAr1a-9NgF?usp=sharing|
| Notebook | https://colab.research.google.com/drive/1wsBOzl2gvDBfXgv9vYB6ZOfFHXHZhN3d?usp=sharing |
| Presentation video | https://youtu.be/aKj-4mTtk2I |
## Problem

Distance-learning universities lose a large share of each cohort to failure and withdrawal, and tutors can only help students they know are struggling. This project asks whether a model can identify, 60 days after a module starts, the enrolled students who will go on to fail or withdraw, using only information the university holds on that day.

The task is binary classification. `at_risk = 1` when the final result is Fail or Withdrawn, and 0 when it is Pass or Distinction.

## Dataset

| | |
|---|---|
| Name | Open University Learning Analytics Dataset (OULAD) |
| Source | UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset> |
| DOI | [10.24432/C5KK69](https://doi.org/10.24432/C5KK69) |
| Licence | CC BY 4.0 |
| Raw size | 32,593 enrolments in `studentInfo.csv`; about 10.6 million rows of clickstream in `studentVle.csv` |
| Tables | `studentInfo`, `studentRegistration`, `courses`, `assessments`, `studentAssessment`, `vle`, `studentVle` |
| Target | `final_result` (Distinction, Pass, Fail, Withdrawn), merged into `at_risk` |

The dataset is **not stored in this repository**. The notebook downloads the 45 MB zip from UCI on first run and extracts it into `data/`, which is git-ignored. A copy of the raw zip is in the Google Drive folder linked above.

## Repository layout

```
.
├── oulad_early_warning.ipynb   Full pipeline: data, EDA, features, ML, DL, evaluation
├── requirements.txt            Python dependencies
├── README.md
├── Interpretations/
│   └── figures/                Every plot from the reported run (the notebook also saves 200 dpi copies to Drive)
├── data/                       Created by the notebook, git-ignored
└── models/                     Created by the notebook, git-ignored
```

## How to run

### Google Colab (recommended)

1. Click the badge above, or open the notebook from GitHub in Colab.
2. Choose **Runtime > Run all**. The first cells download the dataset.
3. Approve the Google Drive permission prompt at the start. Models, figures and the raw dataset zip are saved to `My Drive/oulad_summative`.

A free CPU runtime is enough. The hyperparameter searches take most of the run time, so a full run takes tens of minutes. For a quick end-to-end check, set `FAST_MODE = True` in the configuration cell.

### Locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook oulad_early_warning.ipynb
```

## Pipeline

1. **Cohort.** Enrolments still active on day 60. Students who withdrew earlier are excluded because their outcome is already known.
2. **Feature engineering**, using only events dated on or before day 60:
   * *Background:* module, demographics, prior attempts, credits, registration lead time, concurrent modules.
   * *VLE engagement:* total clicks, active days, weekly clicks, weekly trend and variability, days since last login, clicks by activity type, clicks relative to the cohort median.
   * *Assessment:* submission rate, mean, minimum and weighted score, timeliness.
3. **Split.** About 70 / 15 / 15 train, validation and test, grouped by student so nobody appears in more than one split.
4. **Preprocessing.** Median imputation, `log1p` on click counts, standardisation, one-hot encoding, all inside a scikit-learn `ColumnTransformer` that is fitted on training data only.
5. **Traditional ML (scikit-learn).** Logistic regression, random forest, histogram gradient boosting and a calibrated linear SVM, each tuned with `RandomizedSearchCV` under `StratifiedGroupKFold`.
6. **Deep learning (TensorFlow/Keras).** `tf.data` input pipelines; Sequential MLPs of varying capacity and regularisation; a random search over architecture and optimiser; a Functional-API model with one embedding per categorical feature.
7. **Evaluation.** Accuracy, precision, recall, F1, ROC-AUC and PR-AUC on a held-out test set; confusion matrices; ROC and precision-recall curves; learning curves; bootstrap confidence intervals; calibration.
8. **Error analysis.** Errors by original outcome, by module and by demographic subgroup; permutation importance; a temporal holdout on the final presentation.

## Experiments

| ID | Experiment | Question |
|----|------------|----------|
| ML-1 | Baseline models with defaults | How do the algorithm families compare out of the box? |
| ML-2 | Randomised hyperparameter search | How much does tuning add? |
| ML-3 | Feature-family ablation | What is each group of engineered features worth? |
| ML-4 | Cutoff sensitivity (day 0 to 120) | How early can at-risk students be identified? |
| DL-1 | Small MLP | What does a plain network achieve? |
| DL-2 | Large MLP without regularisation | Does extra capacity help or overfit? |
| DL-3 | Large MLP with dropout, L2 and batch normalisation | Does regularisation close the gap? |
| DL-4 | Random search over architecture and optimiser | Which settings matter? |
| DL-5 | Functional model with categorical embeddings | Do learned embeddings beat one-hot encoding? |

## Results

Held-out test set: 3,971 enrolments, 41.9% at risk. The two finalists are shown at the F1-optimal thresholds chosen on the validation set (0.38 for logistic regression, 0.35 for the MLP). The other rows use a threshold of 0.5.

| Model | Paradigm | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|----------|-----------|--------|----|---------|--------|
| MLP 256-128-64, RMSprop (DL-4 trial 10) | Deep learning | 0.730 | 0.644 | 0.794 | 0.711 | **0.835** | **0.813** |
| Logistic regression (default) | Traditional ML | 0.748 | 0.689 | 0.727 | 0.707 | 0.831 | 0.808 |
| Hist gradient boosting (tuned) | Traditional ML | 0.758 | 0.762 | 0.615 | 0.681 | 0.828 | 0.809 |
| Random forest (tuned) | Traditional ML | 0.758 | 0.773 | 0.598 | 0.674 | 0.827 | 0.806 |
| Linear SVM (tuned) | Traditional ML | 0.757 | 0.790 | 0.573 | 0.664 | 0.822 | 0.798 |
| Dummy (class prior) | Baseline | 0.581 | 0.000 | 0.000 | 0.000 | 0.500 | 0.419 |

* **The two paradigms are statistically tied.** A 95% bootstrap interval for the ROC-AUC difference (traditional minus deep) is [−0.008, +0.000].
* **Behaviour matters far more than background.** Background features alone give a ROC-AUC of 0.649 and assessment features alone 0.810. All features together give 0.836.
* **Earlier predictions are less accurate.** ROC-AUC is 0.721 on day 0, 0.836 on day 60 and 0.894 on day 120.
* **Most missed students looked fine at day 60.** Their median score was 82 and they were active on 27 days, so these errors reflect a limit of the data rather than of the model.
* **The model raises a fairness concern.** False-positive rates are 2.5× higher in the most deprived areas than in the least deprived.

## Saved models

Stored in the Google Drive folder, not in Git.

| File | Contents |
|------|----------|
| `best_traditional_model.joblib` | Best scikit-learn pipeline, preprocessing included |
| `best_deep_model.keras` (and `.h5`) | Best Keras model |
| `deep_model_preprocessing.joblib` | Fitted transformers the Keras model needs |
| `run_metadata.json` | Configuration, thresholds, test metrics, library versions |

```python
import joblib
from tensorflow import keras

traditional = joblib.load("best_traditional_model.joblib")   # traditional.predict_proba(feature_table)
deep = keras.models.load_model("best_deep_model.keras")
```

## Reproducibility

* One seed (`SEED = 42`) is set for Python, NumPy and TensorFlow, and passed to every splitter, search and model.
* The first code cell prints library versions, and `run_metadata.json` records them with the results.
* Neural-network training on a GPU is not bit-for-bit deterministic, so the deep learning scores can differ slightly between runs.

## Limitations

OULAD covers seven modules at one UK distance-learning university in 2013 and 2014, so the findings may not transfer to other institutions or to current online teaching. Clicks are a proxy for study, the at-risk label merges withdrawal with academic failure, and students who leave before day 60 are outside the model's scope. Section 12 of the notebook discusses these in full.

## Citation

J. Kuzilek, M. Hlosta, and Z. Zdrahal, "Open University Learning Analytics dataset," *Scientific Data*, vol. 4, Art. no. 170171, 2017, doi: 10.1038/sdata.2017.171.
