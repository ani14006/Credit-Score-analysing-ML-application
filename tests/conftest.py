import joblib
import pytest

from ml.train import train


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train from the real dataset into a temporary artifact directory."""
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    result = train(artifact_dir=artifact_dir, verbose=False)
    result["artifact_dir"] = artifact_dir
    result["bundle"] = joblib.load(artifact_dir / "model.joblib")
    return result


@pytest.fixture
def sample_customer(trained):
    """First held-out test row, as an API payload."""
    row = trained["X_test"].iloc[0].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
