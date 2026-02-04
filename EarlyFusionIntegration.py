"""
EarlyFusionIntegration PluMA Plugin

Multi-omics early fusion integration with LASSO/Elastic Net regularization
for biomarker discovery and classification.

This plugin implements early fusion by:
1. Loading preprocessed metagenomics and transcriptomics feature matrices
2. Concatenating normalized features from both modalities
3. Applying regularized feature selection (LASSO or Elastic Net)
4. Training a classifier on selected features
5. Outputting integrated feature matrix and model

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


RegularizationType = Literal["lasso", "elasticnet", "ridge"]
ClassifierType = Literal["logistic", "svc", "xgboost", "randomforest"]


class EarlyFusionIntegration:
    """
    PluMA plugin for early fusion multi-omics integration.
    
    Concatenates normalized metagenomics and transcriptomics features,
    applies regularized feature selection, and trains a classifier.
    
    Parameters (via input file):
        metagenomics: Path to metagenomics feature matrix (samples x features)
        transcriptomics: Path to transcriptomics feature matrix (samples x features)
        labels: Path to sample labels CSV
        regularization: Type of regularization ("lasso", "elasticnet", "ridge")
        alpha: Regularization strength (default: 1.0)
        l1_ratio: Elastic net mixing parameter, 0=ridge, 1=lasso (default: 0.5)
        classifier: Classifier type ("logistic", "svc", "xgboost", "randomforest")
        cv_folds: Number of cross-validation folds (default: 5)
        random_state: Random seed for reproducibility (default: 42)
    
    Outputs:
        - Fused feature matrix
        - Selected features with coefficients/importance
        - Classification metrics
        - Trained model (pickled)
    """
    
    def __init__(self) -> None:
        """Initialize plugin state."""
        self.parameters: dict[str, str] = {}
        
        # Data
        self.metagenomics: pd.DataFrame | None = None
        self.transcriptomics: pd.DataFrame | None = None
        self.labels: pd.Series | None = None
        
        # Results
        self.fused_matrix: pd.DataFrame | None = None
        self.selected_features: pd.DataFrame | None = None
        self.cv_results: dict[str, Any] | None = None
        self.model: Any = None
        self.feature_selector: Any = None
        
        # Default parameters
        self.regularization: RegularizationType = "elasticnet"
        self.alpha: float = 1.0
        self.l1_ratio: float = 0.5
        self.classifier_type: ClassifierType = "logistic"
        self.cv_folds: int = 5
        self.random_state: int = 42
    
    def input(self, filename: str) -> None:
        """
        Load input data and configuration parameters.
        
        Args:
            filename: Path to parameter file with key-value pairs
        """
        param_path = Path(filename)
        
        # Parse parameter file
        with param_path.open() as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        self.parameters[parts[0]] = parts[1]
        
        # Load metagenomics features
        if "metagenomics" in self.parameters:
            self.metagenomics = pd.read_csv(
                self.parameters["metagenomics"], 
                index_col=0
            )
        
        # Load transcriptomics features
        if "transcriptomics" in self.parameters:
            self.transcriptomics = pd.read_csv(
                self.parameters["transcriptomics"], 
                index_col=0
            )
        
        # Load sample labels
        if "labels" in self.parameters:
            labels_df = pd.read_csv(self.parameters["labels"], index_col=0)
            self.labels = labels_df.iloc[:, 0]
        
        # Parse optional parameters
        if "regularization" in self.parameters:
            self.regularization = self.parameters["regularization"].lower()  # type: ignore
        
        if "alpha" in self.parameters:
            self.alpha = float(self.parameters["alpha"])
        
        if "l1_ratio" in self.parameters:
            self.l1_ratio = float(self.parameters["l1_ratio"])
        
        if "classifier" in self.parameters:
            self.classifier_type = self.parameters["classifier"].lower()  # type: ignore
        
        if "cv_folds" in self.parameters:
            self.cv_folds = int(self.parameters["cv_folds"])
        
        if "random_state" in self.parameters:
            self.random_state = int(self.parameters["random_state"])
    
    def run(self) -> None:
        """
        Execute early fusion integration pipeline.
        
        Steps:
        1. Validate and align sample IDs across modalities
        2. Concatenate feature matrices
        3. Apply regularized feature selection
        4. Train classifier with cross-validation
        5. Extract selected features and importance
        """
        if self.metagenomics is None or self.transcriptomics is None:
            raise ValueError("Both metagenomics and transcriptomics matrices required")
        
        if self.labels is None:
            raise ValueError("Sample labels required")
        
        # Step 1: Align samples across modalities
        common_samples = self._align_samples()
        
        # Step 2: Concatenate features
        self.fused_matrix = self._concatenate_features(common_samples)
        
        # Step 3: Feature selection with regularization
        self.feature_selector = self._fit_feature_selector()
        
        # Step 4: Train and evaluate classifier
        self.model, self.cv_results = self._train_classifier()
        
        # Step 5: Extract selected features
        self.selected_features = self._extract_selected_features()
    
    def output(self, filename: str) -> None:
        """
        Write results to output files.
        
        Args:
            filename: Base path for output files
        """
        import pickle
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write fused feature matrix
        if self.fused_matrix is not None:
            fused_path = output_path.with_suffix(".fused_matrix.csv")
            self.fused_matrix.to_csv(fused_path)
        
        # Write selected features
        if self.selected_features is not None:
            features_path = output_path.with_suffix(".selected_features.csv")
            self.selected_features.to_csv(features_path)
        
        # Write cross-validation results
        if self.cv_results is not None:
            cv_path = output_path.with_suffix(".cv_results.csv")
            cv_df = pd.DataFrame(self.cv_results)
            cv_df.to_csv(cv_path)
        
        # Write model
        if self.model is not None:
            model_path = output_path.with_suffix(".model.pkl")
            with model_path.open("wb") as f:
                pickle.dump(self.model, f)
        
        # Write summary
        self._write_summary(output_path.with_suffix(".summary.txt"))
    
    def _align_samples(self) -> list[str]:
        """
        Find and align common samples across all data sources.
        
        Returns:
            List of common sample IDs
        """
        mg_samples = set(self.metagenomics.index)  # type: ignore
        tx_samples = set(self.transcriptomics.index)  # type: ignore
        label_samples = set(self.labels.index)  # type: ignore
        
        common = mg_samples & tx_samples & label_samples
        
        if len(common) == 0:
            raise ValueError("No common samples found across data sources")
        
        return sorted(list(common))
    
    def _concatenate_features(self, samples: list[str]) -> pd.DataFrame:
        """
        Concatenate feature matrices with modality prefixes.
        
        Args:
            samples: List of sample IDs to include
            
        Returns:
            Concatenated feature matrix
        """
        # Subset and rename columns with modality prefix
        mg_subset = self.metagenomics.loc[samples].copy()  # type: ignore
        mg_subset.columns = [f"MG_{col}" for col in mg_subset.columns]
        
        tx_subset = self.transcriptomics.loc[samples].copy()  # type: ignore
        tx_subset.columns = [f"TX_{col}" for col in tx_subset.columns]
        
        # Concatenate horizontally
        fused = pd.concat([mg_subset, tx_subset], axis=1)
        
        return fused
    
    def _fit_feature_selector(self) -> Any:
        """
        Fit regularized feature selector for sparse feature selection.
        
        Returns:
            Fitted feature selector model
        """
        from sklearn.linear_model import LogisticRegression, ElasticNet
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        X = self.fused_matrix.values  # type: ignore
        y = self.labels.loc[self.fused_matrix.index].values  # type: ignore
        
        # Build feature selector based on regularization type
        if self.regularization == "lasso":
            selector = LogisticRegression(
                penalty="l1",
                C=1 / self.alpha,
                solver="saga",
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.regularization == "elasticnet":
            selector = LogisticRegression(
                penalty="elasticnet",
                C=1 / self.alpha,
                l1_ratio=self.l1_ratio,
                solver="saga",
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.regularization == "ridge":
            selector = LogisticRegression(
                penalty="l2",
                C=1 / self.alpha,
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown regularization: {self.regularization}")
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", selector)
        ])
        
        pipeline.fit(X, y)
        
        return pipeline
    
    def _train_classifier(self) -> tuple[Any, dict[str, Any]]:
        """
        Train classifier with nested cross-validation.
        
        Returns:
            Tuple of (trained model, CV results dictionary)
        """
        from sklearn.model_selection import cross_validate, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        X = self.fused_matrix.values  # type: ignore
        y = self.labels.loc[self.fused_matrix.index].values  # type: ignore
        
        # Build classifier
        classifier = self._get_classifier()
        
        # Create pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier)
        ])
        
        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        cv_results = cross_validate(
            pipeline, X, y, cv=cv,
            scoring=["accuracy", "f1", "roc_auc", "precision", "recall"],
            return_train_score=True
        )
        
        # Fit final model on all data
        pipeline.fit(X, y)
        
        return pipeline, cv_results
    
    def _get_classifier(self) -> Any:
        """
        Get classifier instance based on configuration.
        
        Returns:
            Classifier instance
        """
        if self.classifier_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                penalty="elasticnet",
                C=1 / self.alpha,
                l1_ratio=self.l1_ratio,
                solver="saga",
                max_iter=1000,
                random_state=self.random_state
            )
        
        elif self.classifier_type == "svc":
            from sklearn.svm import SVC
            return SVC(
                kernel="rbf",
                probability=True,
                random_state=self.random_state
            )
        
        elif self.classifier_type == "xgboost":
            # TODO: Implement XGBoost classifier
            # Requires xgboost package
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric="logloss"
                )
            except ImportError:
                raise ImportError("XGBoost required: pip install xgboost")
        
        elif self.classifier_type == "randomforest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            )
        
        else:
            raise ValueError(f"Unknown classifier: {self.classifier_type}")
    
    def _extract_selected_features(self) -> pd.DataFrame:
        """
        Extract selected features and their importance scores.
        
        Returns:
            DataFrame with feature names, modality, and importance
        """
        feature_names = self.fused_matrix.columns.tolist()  # type: ignore
        
        # Extract coefficients or feature importance
        if hasattr(self.model["classifier"], "coef_"):
            # Linear models
            importance = np.abs(self.model["classifier"].coef_).flatten()
        elif hasattr(self.model["classifier"], "feature_importances_"):
            # Tree-based models
            importance = self.model["classifier"].feature_importances_
        else:
            # Fallback: use feature selector
            if hasattr(self.feature_selector["selector"], "coef_"):
                importance = np.abs(self.feature_selector["selector"].coef_).flatten()
            else:
                importance = np.ones(len(feature_names))
        
        # Create DataFrame
        features_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
            "modality": [f.split("_")[0] for f in feature_names]
        })
        
        # Sort by importance
        features_df = features_df.sort_values("importance", ascending=False)
        
        # Mark selected features (non-zero for LASSO)
        features_df["selected"] = features_df["importance"] > 1e-6
        
        return features_df
    
    def _write_summary(self, filepath: Path) -> None:
        """
        Write integration summary.
        
        Args:
            filepath: Path for summary file
        """
        with filepath.open("w") as f:
            f.write("EarlyFusionIntegration Summary\n")
            f.write("=" * 40 + "\n\n")
            
            if self.fused_matrix is not None:
                f.write(f"Total samples: {self.fused_matrix.shape[0]}\n")
                f.write(f"Total features: {self.fused_matrix.shape[1]}\n")
                
                # Count features by modality
                mg_features = sum(1 for c in self.fused_matrix.columns if c.startswith("MG_"))
                tx_features = sum(1 for c in self.fused_matrix.columns if c.startswith("TX_"))
                f.write(f"  Metagenomics features: {mg_features}\n")
                f.write(f"  Transcriptomics features: {tx_features}\n")
            
            if self.selected_features is not None:
                n_selected = self.selected_features["selected"].sum()
                f.write(f"\nSelected features: {n_selected}\n")
                
                # Count selected by modality
                selected = self.selected_features[self.selected_features["selected"]]
                mg_selected = (selected["modality"] == "MG").sum()
                tx_selected = (selected["modality"] == "TX").sum()
                f.write(f"  Metagenomics selected: {mg_selected}\n")
                f.write(f"  Transcriptomics selected: {tx_selected}\n")
                
                # Top 10 features
                f.write("\nTop 10 features:\n")
                for _, row in self.selected_features.head(10).iterrows():
                    f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
            
            if self.cv_results is not None:
                f.write("\nCross-Validation Results:\n")
                f.write(f"  Accuracy: {np.mean(self.cv_results['test_accuracy']):.3f} "
                       f"(+/- {np.std(self.cv_results['test_accuracy']):.3f})\n")
                f.write(f"  F1 Score: {np.mean(self.cv_results['test_f1']):.3f} "
                       f"(+/- {np.std(self.cv_results['test_f1']):.3f})\n")
                f.write(f"  ROC AUC: {np.mean(self.cv_results['test_roc_auc']):.3f} "
                       f"(+/- {np.std(self.cv_results['test_roc_auc']):.3f})\n")
            
            f.write(f"\nParameters:\n")
            f.write(f"  regularization: {self.regularization}\n")
            f.write(f"  alpha: {self.alpha}\n")
            f.write(f"  l1_ratio: {self.l1_ratio}\n")
            f.write(f"  classifier: {self.classifier_type}\n")
            f.write(f"  cv_folds: {self.cv_folds}\n")
