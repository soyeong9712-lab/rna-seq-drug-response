import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from src.utils.paths import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR
from src.utils.logger import get_logger
from src.features.preprocess import load_processed, log2_transform, filter_low_expression

# ✅ 시각화(Seaborn 없이 matplotlib만 쓰도록 plot_expression.py도 아래에 같이 수정해줄 것)
from src.visualization.plot_expression import (
    plot_heatmap,
    plot_pca,
    plot_pca_with_variance,          # ✅ [추가] explained variance 포함 PCA
    plot_sample_correlation_heatmap, # ✅ [추가] 샘플 상관관계 heatmap
    plot_volcano,                   # ✅ [추가] volcano plot
    plot_topgene_boxplots           # ✅ [추가] top gene boxplot
)

logger = get_logger()

def build_model(random_state: int = 42):
    """
    Pipeline:
      log/필터는 미리 적용(데이터프레임 처리)
      scaler -> SelectFromModel(ElasticNet LR) -> Voting(RF, LR)
    """

    # 1) 피처 선택용 LR(Elastic Net)
    selector_lr = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=0.5,
        max_iter=15000,
        random_state=random_state
    )

    # 2) 최종 분류기들
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=random_state
    )

    lr_final = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=1.0,
        max_iter=3000,
        random_state=random_state
    )

    voter = VotingClassifier(
        estimators=[("rf", rf), ("lr", lr_final)],
        voting="soft"
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("select", SelectFromModel(estimator=selector_lr, max_features=300)),
        ("clf", voter)
    ])

    return pipe


def _sanitize_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ [추가] log 변환/필터링 과정에서 생길 수 있는 inf/NaN 방어
    - PCA/상관/volcano/모델 학습이 터지는 흔한 원인 제거
    """
    X2 = X.replace([np.inf, -np.inf], np.nan)

    # 유전자(컬럼) 단위로 NaN이 섞인 컬럼 제거 (가장 안전)
    before = X2.shape[1]
    X2 = X2.dropna(axis=1)
    after = X2.shape[1]
    if after != before:
        logger.warning(f"Dropped {before - after} genes containing NaN/Inf after log transform.")
    return X2


def _get_selector_fitted_estimator(selector: SelectFromModel):
    """
    ✅ [추가] SelectFromModel 내부 fitted estimator 안전하게 가져오기
    - selector.estimator_가 있으면 그걸 사용 (fit 후에 생김)
    - 없으면 estimator를 fallback (이 경우 coef_ 없을 수 있으니 에러)
    """
    if hasattr(selector, "estimator_"):
        return selector.estimator_
    return selector.estimator


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x_path = PROCESSED_DIR / "X_gene_expression.csv"
    y_path = PROCESSED_DIR / "y_labels.csv"

    # 1) 만약 processed가 없으면 생성부터
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "processed 데이터가 없습니다. 먼저 아래를 실행하세요:\n"
            "python -m src.data.make_dataset"
        )

    # 2) 로드
    X, y = load_processed(str(x_path), str(y_path))
    logger.info(f"Loaded X shape (samples x genes): {X.shape}")
    logger.info(f"y:\n{y.value_counts()}")

    # 3) 전처리(데이터프레임 레벨에서 수행)
    X2 = X.copy()

    # log2 변환
    X2 = log2_transform(X2)

    # ✅ [추가] inf/NaN 방어(특히 log 변환 후)
    X2 = _sanitize_matrix(X2)

    # low expression filter
    X2_t = filter_low_expression(X2.T, min_nonzero_frac=0.34)  # genes x samples
    X2 = X2_t.T  # back to samples x genes

    # ✅ [추가] 필터 결과가 0 genes이면 이후 전부 터짐 → 즉시 중단 + 가이드
    if X2.shape[1] == 0:
        raise ValueError(
            "After filter_low_expression, no genes remain (X2 has 0 columns).\n"
            "👉 min_nonzero_frac 값을 낮추거나(예: 0.2~0.3),\n"
            "👉 filter_low_expression 함수에서 axis 방향(genes/samples)을 확인하세요."
        )

    # ✅ [추가] 다시 한 번 방어(필터 후에도 NaN 생길 수 있으면 제거)
    X2 = _sanitize_matrix(X2)

    logger.info(f"After filter: {X2.shape}")

    # ✅ [추가 1] 여기서 PCA 먼저 뽑아두면 "전처리 후 샘플 분리" 그림을 바로 만들 수 있음
    #    - 보고서/수행내역서에 넣기 좋은 그림
    #    - 파일 저장 경로는 figures/ 폴더 추천 (없으면 mkdir)

    from pathlib import Path

    # ✅ [수정] 실행 위치가 달라도 reports와 같은 레벨에 figures를 만들도록
    figures_dir = REPORTS_DIR.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # PCA 시각화 (샘플 분리 확인용)
    # (기존) plot_pca(X2, y, out_path=figures_dir / "pca_samples.png")

    # ✅✅ [추가 1-1] PCA + Explained Variance(PC1/PC2가 몇 % 설명하는지)
    # - 수행내역서에서 “PC1이 전체 변동의 XX%를 설명” 문장 넣기 쉬움
    plot_pca_with_variance(X2, y, out_path=figures_dir / "pca_samples.png")

    # ✅✅ [추가 1-2] Sample-to-Sample Correlation Heatmap
    # - RNA-seq QC 스타일 그림(재현성/클러스터링 근거)
    plot_sample_correlation_heatmap(X2, y, out_path=figures_dir / "sample_correlation.png")

    # ✅✅ [추가 1-3] Volcano Plot (Control vs Treated)
    # - RNA-seq 레포트 “국룰” 그림
    # - 샘플이 적어서 p-value는 참고용(보조 분석)으로 설명하면 됨
    # ✅ [주의] plot_volcano 내부에서 그룹 샘플 수 체크(>=2) 없으면 여기서 터질 수 있음
    plot_volcano(
        X2,
        y,
        out_path=figures_dir / "volcano.png",
        # 필요하면 컷오프 조절 가능
        log2fc_thr=1.0,
        p_thr=0.05
    )

    # 4) 모델 + LOOCV 예측
    model = build_model()
    cv = LeaveOneOut()

    proba = cross_val_predict(model, X2, y, cv=cv, method="predict_proba")

    # ✅ [추가] 클래스 불균형/폴드 문제로 proba가 (n,1) 나오는 경우 방어
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise RuntimeError(
            f"predict_proba returned shape {proba.shape}. "
            "LOOCV fold에서 한 클래스만 남았을 수 있습니다(표본/불균형 이슈)."
        )

    y_pred = (proba[:, 1] >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, proba[:, 1])
    except Exception:
        auc = None

    cm = confusion_matrix(y, y_pred)
    rep = classification_report(y, y_pred, digits=4)

    logger.info(f"LOOCV Accuracy: {acc:.4f}")
    if auc is not None:
        logger.info(f"LOOCV ROC-AUC : {auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Report:\n{rep}")

    # 5) 전체 데이터로 최종 학습(모델 저장)
    model.fit(X2, y)

    import joblib
    model_path = MODELS_DIR / "voting_rf_lr.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Saved model: {model_path}")

    # 6) 선택된 유전자(최종 학습 기준) 뽑기
    selector = model.named_steps["select"]
    support = selector.get_support()
    selected_genes = X2.columns[support].tolist()

    # ✅ [수정] SelectFromModel 내부 fitted estimator에서 coef_ 안전하게 가져오기
    sel_lr = _get_selector_fitted_estimator(selector)
    if not hasattr(sel_lr, "coef_"):
        raise RuntimeError(
            "Selector LR has no coef_. "
            "SelectFromModel 내부 estimator가 제대로 fit되지 않았습니다."
        )

    coefs = np.asarray(sel_lr.coef_).ravel()

    # ✅ [추가] support 길이와 coef 길이가 안 맞으면 바로 에러(디버깅용)
    if support.shape[0] != coefs.shape[0]:
        raise RuntimeError(
            f"Mismatch: support({support.shape[0]}) vs coef({coefs.shape[0]}). "
            "filter/columns alignment을 확인하세요."
        )

    selected_coefs = coefs[support]
    importance = np.abs(selected_coefs)

    genes_df = pd.DataFrame({
        "gene": selected_genes,
        "coef": selected_coefs,
        "abs_coef": importance
    }).sort_values("abs_coef", ascending=False)

    top_path = REPORTS_DIR / "selected_genes_top.csv"
    genes_df.to_csv(top_path, index=False)
    logger.info(f"Saved selected genes: {top_path} (n={len(genes_df)})")

    # ✅✅ [추가 3 - 핵심] 여기서 "top 유전자 리스트"가 생겼으니
    #     이걸로 heatmap을 만들면 수행내역서에 바로 넣을 수 있는 결과 그림이 됨
    #
    #  - 추천: abs_coef 상위 30~50개
    #  - plot_heatmap은 X2(샘플x유전자), y(라벨), genes(list), out_path를 받도록 구현돼 있어야 함
    #
    # Heatmap (상위 30개 유전자)
    top_genes = genes_df["gene"].head(30).tolist()
    plot_heatmap(X2, y, top_genes, out_path=figures_dir / "heatmap_top30.png")

    # ✅ [추가 4 - 선택] PCA도 "선택된 유전자만"으로 다시 그리면 더 깔끔함
    plot_pca_with_variance(X2[top_genes], y, out_path=figures_dir / "pca_topgenes.png")

    # ✅✅ [추가 5 - 2순위] Top gene Boxplot (핵심 유전자 몇 개만)
    # - Heatmap은 “전체 패턴”, Boxplot은 “개별 유전자 설명”
    # - 수행내역서에서 “대표 유전자 발현 비교” 그림으로 쓰기 좋음
    plot_topgene_boxplots(
        X2,
        y,
        genes=genes_df["gene"].head(6).tolist(),   # 상위 6개만(그림이 과해지면 4개로)
        out_path=figures_dir / "topgenes_boxplot.png"
    )

    # 7) 메트릭 저장
    metrics = {
        "n_samples": int(X2.shape[0]),
        "n_genes_before": int(X.shape[1]),
        "n_genes_after_filter": int(X2.shape[1]),
        "n_selected_genes": int(len(selected_genes)),
        "loocv_accuracy": float(acc),
        "loocv_auc": (float(auc) if auc is not None else None),
        "confusion_matrix": cm.tolist()
    }

    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
