# Reproducible result summaries

This directory keeps lightweight, reviewable outputs for the two retained CNN+GNN experiments. Large checkpoints, datasets, embeddings, per-image predictions and generated working directories stay under the ignored `results/`, `.cache/` and `data/` paths.

## ALL-IDB1

`allidb1_inception_v3_dynamic_gnn/` contains the completed five-fold duplicate-aware image-level evaluation: 98.15% accuracy, 98.13% balanced accuracy and 0.9941 macro ROC-AUC. The evaluation protocol is stated in the notebook and must accompany the result.

## C-NMC

`cnmc_se_inception_v3_gnn_3fold/` contains the completed three-fold SE-InceptionV3 + Dynamic GNN ensemble evaluated on the official preliminary test set: 78.47% accuracy, 73.68% balanced accuracy and 0.8621 macro ROC-AUC. This result is preserved as measured; it did not reach the 93% target.
