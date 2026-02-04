# EarlyFusionIntegration

A PluMA plugin for multi-omics early fusion integration with LASSO/Elastic Net regularization for biomarker discovery and classification.

## Overview

This plugin implements early fusion by:

1. Loading preprocessed metagenomics and transcriptomics feature matrices
2. Concatenating normalized features from both modalities
3. Applying regularized feature selection (LASSO or Elastic Net)
4. Training a classifier on selected features
5. Outputting integrated feature matrix and model

## Installation

```bash
# Clone the repository
git clone https://github.com/quinnjr/EarlyFusionIntegration.git

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- numpy
- pandas
- scikit-learn
- xgboost (optional)

## Usage

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `metagenomics` | Path to metagenomics feature matrix (samples x features) | Required |
| `transcriptomics` | Path to transcriptomics feature matrix (samples x features) | Required |
| `labels` | Path to sample labels CSV | Required |
| `regularization` | Type: `lasso`, `elasticnet`, `ridge` | `elasticnet` |
| `alpha` | Regularization strength | `1.0` |
| `l1_ratio` | Elastic net mixing (0=ridge, 1=lasso) | `0.5` |
| `classifier` | Type: `logistic`, `svc`, `xgboost`, `randomforest` | `logistic` |
| `cv_folds` | Number of cross-validation folds | `5` |
| `random_state` | Random seed for reproducibility | `42` |

### Example Parameter File

```
metagenomics        data/metagenomics_normalized.csv
transcriptomics     data/transcriptomics_normalized.csv
labels              data/sample_labels.csv
regularization      elasticnet
alpha               1.0
l1_ratio            0.5
classifier          randomforest
cv_folds            5
random_state        42
```

### Outputs

- Fused feature matrix
- Selected features list
- Trained classification model
- Cross-validation performance metrics

## Methods

### Early Fusion

Early fusion concatenates feature matrices from multiple omics modalities into a single feature space before applying machine learning. This approach:
- Captures cross-modal feature interactions
- Enables joint feature selection across modalities
- Requires careful normalization to balance modality contributions

### Regularization

- **LASSO (L1)**: Sparse feature selection, drives coefficients to zero
- **Ridge (L2)**: Shrinks coefficients, retains all features
- **Elastic Net**: Combines L1 and L2, balances sparsity and stability

### Classifiers

- **Logistic Regression**: Linear classifier with probabilistic output
- **SVM (SVC)**: Maximum margin classifier
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosted trees

## References

### Multi-omics Integration

1. **Ritchie MD, Holzinger ER, Li R, Pendergrass SA, Kim D** (2015). Methods of integrating data to uncover genotype-phenotype interactions. *Nature Reviews Genetics*, 16(2):85-97. [doi:10.1038/nrg3868](https://doi.org/10.1038/nrg3868)
   - *Comprehensive review of multi-omics integration strategies*

2. **Huang S, Chaudhary K, Garmire LX** (2017). More Is Better: Recent Progress in Multi-Omics Data Integration Methods. *Frontiers in Genetics*, 8:84. [doi:10.3389/fgene.2017.00084](https://doi.org/10.3389/fgene.2017.00084)
   - *Early fusion approaches and comparison*

### Regularization Methods

3. **Tibshirani R** (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society B*, 58(1):267-288. [doi:10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
   - *Original LASSO paper*

4. **Zou H, Hastie T** (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society B*, 67(2):301-320. [doi:10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
   - *Elastic Net methodology*

5. **Hoerl AE, Kennard RW** (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1):55-67. [doi:10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
   - *Ridge regression foundation*

### Classification Methods

6. **Breiman L** (2001). Random Forests. *Machine Learning*, 45:5-32. [doi:10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
   - *Random Forest algorithm*

7. **Chen T, Guestrin C** (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*. [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
   - *XGBoost implementation*

## License

MIT License

## Author

Joseph R. Quinn <quinn.josephr@protonmail.com>
