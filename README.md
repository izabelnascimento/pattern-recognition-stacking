# Pattern Recognition with Stacking

A Pattern Recognition course project that compares five classifiers on ten public datasets. The experiment measures whether a stacking ensemble improves classification quality and how much training time it adds.

- [Project presentation (PowerPoint)](https://docs.google.com/presentation/d/11cno2w9LME9wclFXbtg6OMuGUUvDB6gU/edit?rtpof=true)
- [Experiment architecture (Lucidchart)](https://lucid.app/lucidchart/d013aff7-1d38-4d0b-95bd-a5e771e62c75/edit?page=jJZHimtNuE0X0)
- [Stacking architecture in detail (Lucidchart)](https://lucid.app/lucidchart/d013aff7-1d38-4d0b-95bd-a5e771e62c75/edit?page=JltUdxtw4ZL40)

## Architecture

The notebook loads three datasets from scikit-learn and seven from local CSV files. Each dataset uses the same ten stratified outer folds for all five classifiers: K-nearest neighbors (KNN), an RBF support vector machine (SVM-RBF), Random Forest, a multilayer perceptron (MLP), and stacking. The KNN, SVM-RBF, and MLP pipelines fit feature standardization on their training data.

The stacking ensemble has two levels:

1. **Base classifiers.** KNN, calibrated SVM-RBF, and Random Forest produce class probabilities. Five stratified inner folds within each outer training set generate out-of-fold probabilities for the meta-classifier.
2. **Meta-classifier.** Logistic regression learns to combine those probabilities. The base classifiers are also fitted on the full outer training set. At prediction time, their probabilities on the outer test set become inputs to the fitted logistic regression.

The MLP is evaluated independently. The experiment records nine quality metrics and fit time across 500 outer-fold evaluations (10 datasets × 5 models × 10 folds).

## Datasets

New Thyroid, Wine, Breast Cancer, Digits, Pima Diabetes, Sonar, Ionosphere, Banknote, Seeds, and Phoneme.

## Results

The primary metric is macro F1, averaged first across the ten outer folds and then across datasets.

| Classifier | Mean macro F1 |
| --- | ---: |
| Stacking | **0.9232** |
| Random Forest | 0.9168 |
| MLP | 0.9099 |
| SVM-RBF | 0.9092 |
| KNN | 0.8905 |

The Friedman test detected an overall difference in macro F1 (*p* ≈ 0.008). At the 5% significance level, the Nemenyi post-hoc test identified stacking versus KNN as the only significant pair (*p* ≈ 0.002). Stacking also had the highest average training time per fold in the recorded run.

## Run the experiment

Open [`ReconhecimentoPadroes.ipynb`](ReconhecimentoPadroes.ipynb) in Jupyter and run all cells from the repository root so the notebook can find `dados/`. The first cell installs the required Python packages. The experiment uses seed `42` and writes its results to `saida/`.

| Path | Contents |
| --- | --- |
| [`ReconhecimentoPadroes.ipynb`](ReconhecimentoPadroes.ipynb) | Data loading, model training, evaluation, hypothesis tests, and report generation |
| [`dados/`](dados/) | Seven public datasets stored as CSV files |
| [`saida/resumo_geral.csv`](saida/resumo_geral.csv) | Metric and fit-time averages by classifier |
| [`saida/resultados_por_particao.csv`](saida/resultados_por_particao.csv) | Results for the 500 outer-fold evaluations |
| [`saida/relatorio_comparacao_modelos.pdf`](saida/relatorio_comparacao_modelos.pdf) | Full report, including plots and statistical analysis |
