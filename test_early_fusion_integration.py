"""
Unit tests for EarlyFusionIntegration PluMA Plugin.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from EarlyFusionIntegration import EarlyFusionIntegration


@pytest.fixture
def sample_metagenomics() -> pd.DataFrame:
    """Create sample metagenomics feature matrix."""
    np.random.seed(42)
    n_samples = 50
    n_features = 30
    
    data = np.random.rand(n_samples, n_features)
    sample_names = [f"Sample_{i:03d}" for i in range(n_samples)]
    feature_names = [f"Taxa_{i}" for i in range(n_features)]
    
    return pd.DataFrame(data, index=sample_names, columns=feature_names)


@pytest.fixture
def sample_transcriptomics() -> pd.DataFrame:
    """Create sample transcriptomics feature matrix."""
    np.random.seed(43)
    n_samples = 50
    n_features = 40
    
    data = np.random.rand(n_samples, n_features)
    sample_names = [f"Sample_{i:03d}" for i in range(n_samples)]
    feature_names = [f"Gene_{i}" for i in range(n_features)]
    
    return pd.DataFrame(data, index=sample_names, columns=feature_names)


@pytest.fixture
def sample_labels() -> pd.DataFrame:
    """Create sample labels."""
    sample_names = [f"Sample_{i:03d}" for i in range(50)]
    labels = [0] * 25 + [1] * 25
    
    return pd.DataFrame({"label": labels}, index=sample_names)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def plugin_with_data(temp_dir, sample_metagenomics, sample_transcriptomics, sample_labels):
    """Create plugin instance with loaded test data."""
    # Write test files
    mg_path = temp_dir / "metagenomics.csv"
    tx_path = temp_dir / "transcriptomics.csv"
    labels_path = temp_dir / "labels.csv"
    
    sample_metagenomics.to_csv(mg_path)
    sample_transcriptomics.to_csv(tx_path)
    sample_labels.to_csv(labels_path)
    
    # Write parameter file
    param_path = temp_dir / "params.txt"
    param_path.write_text(f"""metagenomics\t{mg_path}
transcriptomics\t{tx_path}
labels\t{labels_path}
regularization\telasticnet
alpha\t1.0
l1_ratio\t0.5
classifier\tlogistic
cv_folds\t3
random_state\t42
""")
    
    plugin = EarlyFusionIntegration()
    plugin.input(str(param_path))
    
    return plugin


class TestEarlyFusionInit:
    """Tests for plugin initialization."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        plugin = EarlyFusionIntegration()
        
        assert plugin.regularization == "elasticnet"
        assert plugin.alpha == 1.0
        assert plugin.l1_ratio == 0.5
        assert plugin.classifier_type == "logistic"
        assert plugin.cv_folds == 5
        assert plugin.random_state == 42
    
    def test_load_metagenomics(self, temp_dir, sample_metagenomics):
        """Test loading metagenomics data."""
        mg_path = temp_dir / "mg.csv"
        sample_metagenomics.to_csv(mg_path)
        
        param_path = temp_dir / "params.txt"
        param_path.write_text(f"metagenomics\t{mg_path}\n")
        
        plugin = EarlyFusionIntegration()
        plugin.input(str(param_path))
        
        assert plugin.metagenomics is not None
        assert plugin.metagenomics.shape == sample_metagenomics.shape
    
    def test_parse_regularization(self, temp_dir):
        """Test parsing regularization parameter."""
        param_path = temp_dir / "params.txt"
        param_path.write_text("regularization\tlasso\nalpha\t0.5\n")
        
        plugin = EarlyFusionIntegration()
        plugin.input(str(param_path))
        
        assert plugin.regularization == "lasso"
        assert plugin.alpha == 0.5


class TestSampleAlignment:
    """Tests for sample alignment across modalities."""
    
    def test_align_samples_common(self, plugin_with_data):
        """Test finding common samples."""
        plugin = plugin_with_data
        
        common = plugin._align_samples()
        
        assert len(common) == 50
        assert all(s in plugin.metagenomics.index for s in common)
        assert all(s in plugin.transcriptomics.index for s in common)
        assert all(s in plugin.labels.index for s in common)
    
    def test_align_samples_partial_overlap(self, temp_dir):
        """Test alignment with partial sample overlap."""
        # Create data with partial overlap
        mg = pd.DataFrame(
            np.random.rand(10, 5),
            index=[f"S{i}" for i in range(10)],
            columns=[f"F{i}" for i in range(5)]
        )
        tx = pd.DataFrame(
            np.random.rand(10, 5),
            index=[f"S{i}" for i in range(5, 15)],  # Overlap: S5-S9
            columns=[f"G{i}" for i in range(5)]
        )
        labels = pd.DataFrame(
            {"label": [0]*5 + [1]*5},
            index=[f"S{i}" for i in range(5, 15)]
        )
        
        mg.to_csv(temp_dir / "mg.csv")
        tx.to_csv(temp_dir / "tx.csv")
        labels.to_csv(temp_dir / "labels.csv")
        
        param_path = temp_dir / "params.txt"
        param_path.write_text(f"""metagenomics\t{temp_dir}/mg.csv
transcriptomics\t{temp_dir}/tx.csv
labels\t{temp_dir}/labels.csv
""")
        
        plugin = EarlyFusionIntegration()
        plugin.input(str(param_path))
        
        common = plugin._align_samples()
        
        assert len(common) == 5  # S5-S9


class TestFeatureConcatenation:
    """Tests for feature matrix concatenation."""
    
    def test_concatenate_adds_prefixes(self, plugin_with_data):
        """Test that concatenation adds modality prefixes."""
        plugin = plugin_with_data
        samples = plugin._align_samples()
        
        fused = plugin._concatenate_features(samples)
        
        mg_cols = [c for c in fused.columns if c.startswith("MG_")]
        tx_cols = [c for c in fused.columns if c.startswith("TX_")]
        
        assert len(mg_cols) == 30  # Original metagenomics features
        assert len(tx_cols) == 40  # Original transcriptomics features
    
    def test_concatenate_preserves_samples(self, plugin_with_data):
        """Test that concatenation preserves sample count."""
        plugin = plugin_with_data
        samples = plugin._align_samples()
        
        fused = plugin._concatenate_features(samples)
        
        assert len(fused) == len(samples)
    
    def test_concatenate_total_features(self, plugin_with_data):
        """Test total feature count after concatenation."""
        plugin = plugin_with_data
        samples = plugin._align_samples()
        
        fused = plugin._concatenate_features(samples)
        
        expected_features = plugin.metagenomics.shape[1] + plugin.transcriptomics.shape[1]
        assert fused.shape[1] == expected_features


class TestFeatureSelection:
    """Tests for regularized feature selection."""
    
    def test_lasso_selector(self, plugin_with_data):
        """Test LASSO feature selector."""
        plugin = plugin_with_data
        plugin.regularization = "lasso"
        
        samples = plugin._align_samples()
        plugin.fused_matrix = plugin._concatenate_features(samples)
        
        selector = plugin._fit_feature_selector()
        
        assert selector is not None
        assert hasattr(selector, "predict")
    
    def test_elasticnet_selector(self, plugin_with_data):
        """Test Elastic Net feature selector."""
        plugin = plugin_with_data
        plugin.regularization = "elasticnet"
        
        samples = plugin._align_samples()
        plugin.fused_matrix = plugin._concatenate_features(samples)
        
        selector = plugin._fit_feature_selector()
        
        assert selector is not None


class TestClassifiers:
    """Tests for different classifier types."""
    
    def test_logistic_classifier(self, plugin_with_data):
        """Test logistic regression classifier."""
        plugin = plugin_with_data
        plugin.classifier_type = "logistic"
        
        classifier = plugin._get_classifier()
        
        assert classifier is not None
        assert hasattr(classifier, "fit")
        assert hasattr(classifier, "predict")
    
    def test_svc_classifier(self, plugin_with_data):
        """Test SVC classifier."""
        plugin = plugin_with_data
        plugin.classifier_type = "svc"
        
        classifier = plugin._get_classifier()
        
        assert classifier is not None
    
    def test_randomforest_classifier(self, plugin_with_data):
        """Test Random Forest classifier."""
        plugin = plugin_with_data
        plugin.classifier_type = "randomforest"
        
        classifier = plugin._get_classifier()
        
        assert classifier is not None
    
    def test_invalid_classifier(self, plugin_with_data):
        """Test invalid classifier type raises error."""
        plugin = plugin_with_data
        plugin.classifier_type = "invalid"
        
        with pytest.raises(ValueError):
            plugin._get_classifier()


class TestRunPipeline:
    """Tests for full pipeline execution."""
    
    def test_run_completes(self, plugin_with_data):
        """Test that run() completes without error."""
        plugin = plugin_with_data
        
        plugin.run()
        
        assert plugin.fused_matrix is not None
        assert plugin.model is not None
        assert plugin.cv_results is not None
        assert plugin.selected_features is not None
    
    def test_cv_results_structure(self, plugin_with_data):
        """Test CV results have expected structure."""
        plugin = plugin_with_data
        plugin.run()
        
        assert "test_accuracy" in plugin.cv_results
        assert "test_f1" in plugin.cv_results
        assert "test_roc_auc" in plugin.cv_results
    
    def test_selected_features_structure(self, plugin_with_data):
        """Test selected features DataFrame structure."""
        plugin = plugin_with_data
        plugin.run()
        
        assert "feature" in plugin.selected_features.columns
        assert "importance" in plugin.selected_features.columns
        assert "modality" in plugin.selected_features.columns


class TestOutput:
    """Tests for output generation."""
    
    def test_output_creates_files(self, plugin_with_data, temp_dir):
        """Test that output creates expected files."""
        plugin = plugin_with_data
        plugin.run()
        
        output_path = temp_dir / "output"
        plugin.output(str(output_path))
        
        assert output_path.with_suffix(".fused_matrix.csv").exists()
        assert output_path.with_suffix(".selected_features.csv").exists()
        assert output_path.with_suffix(".cv_results.csv").exists()
        assert output_path.with_suffix(".model.pkl").exists()
        assert output_path.with_suffix(".summary.txt").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
