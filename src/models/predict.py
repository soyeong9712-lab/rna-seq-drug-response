import pandas as pd
import joblib

from src.utils.paths import MODELS_DIR, PROCESSED_DIR
from src.features.preprocess import load_processed, log2_transform, filter_low_expression
from src.utils.logger import get_logger

# 예측 스크립트(옵션)

logger = get_logger()

def main():
    model_path = MODELS_DIR / "voting_rf_lr.joblib"
    if not model_path.exists():
        raise FileNotFoundError("모델이 없습니다. 먼저 train을 실행하세요.")

    model = joblib.load(model_path)

    X, y = load_processed(
        str(PROCESSED_DIR / "X_gene_expression.csv"),
        str(PROCESSED_DIR / "y_labels.csv")
    )

    X2 = log2_transform(X)
    X2_t = filter_low_expression(X2.T, min_nonzero_frac=0.34)
    X2 = X2_t.T

    proba = model.predict_proba(X2)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = pd.DataFrame({
        "sample": X2.index,
        "pred": pred,
        "proba_treated": proba,
        "true": y.values
    })

    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
