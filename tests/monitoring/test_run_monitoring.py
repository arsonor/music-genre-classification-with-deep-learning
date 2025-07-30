import os
import pandas as pd
from monitoring.run_monitoring import metrics

def test_metrics_returns_text(tmp_path):
    # Create fake parquet files
    ref_df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    cur_df = pd.DataFrame({"feature1": [1, 2], "feature2": [4, 5]})

    ref_df.to_parquet(tmp_path / "reference.parquet")
    cur_df.to_parquet(tmp_path / "current.parquet")

    os.chdir(tmp_path)
    result = metrics()
    assert isinstance(result.data, bytes)
    assert b"# TYPE" in result.data
