import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils.paths import PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR
from src.utils.logger import get_logger
from src.features.preprocess import load_processed
from src.features.transformers import Log2Transformer, LowExpressionFilter

logger = get_logger()

RANDOM_STATE = 42
N_PERMUTATION = 100  # 보고서용으로 충분


# -------------------------
# permutation histogram plot
# -------------------------
def plot_permutation_histogram(real_acc, perm_accs, model_name, out_path):
    """
    real_acc : float (실제 LOOCV accuracy)
    perm_accs: list or np.array (permutation accuracies)
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(perm_accs, bins=10, alpha=0.7, color="gray", edgecolor="black")
    ax.axvline(
        real_acc,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Real accuracy = {real_acc:.2f}",
    )

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation test ({model_name})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -------------------------
# 모델 정의
# -------------------------
def get_models():
    return {
        "Logistic_L2": LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        # 필요하면 아래 주석 해제
        # "Logistic_ElasticNet": LogisticRegression(
        #     penalty="elasticnet",
        #     solver="saga",
        #     l1_ratio=0.5,
        #     C=0.5,
        #     max_iter=15000,
        #     random_state=RANDOM_STATE
        # ),
        # "LinearSVM": LinearSVC(
        #     C=1.0, dual=False, max_iter=5000, random_state=RANDOM_STATE
        # ),
        # "RandomForest": RandomForestClassifier(
        #     n_estimators=300, random_state=RANDOM_STATE
        # ),
    }


def build_pipeline(model):
    steps = [
        ("log2", Log2Transformer()),
        ("filter", LowExpressionFilter(min_nonzero_frac=0.34)),
        ("scaler", StandardScaler()),
    ]

    # ✅ feature selection은 Logistic에만 적용
    # (주의) 여기서 SelectFromModel에 넣는 model은 "selector용"으로 쓰고,
    # 마지막 분류기는 별도 Logistic으로 둠(지금 네 의도 유지)
    if isinstance(model, LogisticRegression):
        selector = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )
        steps.append(("select", SelectFromModel(selector, max_features=300)))

        final_clf = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )
        steps.append(("clf", final_clf))
    else:
        steps.append(("clf", model))

    return Pipeline(steps)


# -------------------------
# permutation test
# -------------------------
def permutation_test(pipe, X, y, n_perm=200, random_state=42):
    cv = LeaveOneOut()
    rng = np.random.default_rng(random_state)

    scores = []
    y_arr = y.values

    for _ in range(n_perm):
        y_perm = rng.permutation(y_arr)
        pred = cross_val_predict(pipe, X, y_perm, cv=cv)
        scores.append(accuracy_score(y_perm, pred))

    return np.array(scores)


# -------------------------
# main
# -------------------------
def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_processed(
        str(PROCESSED_DIR / "X_gene_expression.csv"),
        str(PROCESSED_DIR / "y_labels.csv"),
    )

    results = []

    for name, model in get_models().items():
        logger.info(f"=== {name} ===")

        pipe = build_pipeline(model)
        cv = LeaveOneOut()

        # LOOCV 예측
        if hasattr(model, "predict_proba"):
            proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
            y_pred = (proba >= 0.5).astype(int)
            auc = roc_auc_score(y, proba)
        else:
            y_pred = cross_val_predict(pipe, X, y, cv=cv)
            auc = None

        acc = accuracy_score(y, y_pred)

        # permutation test
        perm_scores = permutation_test(
            pipe, X, y, n_perm=N_PERMUTATION, random_state=RANDOM_STATE
        )
        p_value = float((perm_scores >= acc).mean())

        results.append(
            {
                "model": name,
                "loocv_accuracy": float(acc),
                "loocv_auc": (float(auc) if auc is not None else None),
                "perm_mean_acc": float(perm_scores.mean()),
                "perm_std_acc": float(perm_scores.std()),
                "perm_p_value": p_value,
            }
        )

        logger.info(
            f"ACC={acc:.3f}, "
            f"perm_mean={perm_scores.mean():.3f}, "
            f"p={p_value:.4f}"
        )

        # ✅✅ 히스토그램 저장 (원하면 Logistic_L2만 저장)
        if name == "Logistic_L2":
            out_fig = FIGURES_DIR / "permutation_hist_logistic_l2.png"
            plot_permutation_histogram(
                real_acc=acc,
                perm_accs=perm_scores,
                model_name=name,
                out_path=out_fig,
            )
            logger.info(f"Saved permutation histogram: {out_fig}")

    df = pd.DataFrame(results).sort_values("loocv_accuracy", ascending=False)

    out_csv = REPORTS_DIR / "model_comparison_permutation.csv"
    df.to_csv(out_csv, index=False)

    logger.info(f"Saved: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
