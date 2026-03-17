import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

# (프로젝트 내부 모듈) 사용하려면 PYTHONPATH가 잡혀야 해서
# 실행할 때 "python -m streamlit run app.py" 방식 말고
# 그냥 "streamlit run app.py"로도 되게끔 루트 기준 import는 피하고
# 파일만 직접 읽는 방식으로 대시보드 구성하는 게 제일 안전함.

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ✅ (추가) 최근 예측 결과를 "세션만"이 아니라 "파일로도" 저장해두면
# - 사이드바 이동/리런/일부 상황에서도 안정적으로 Overview에서 불러올 수 있음
# - (옵션) 서버 재시작해도 마지막 결과를 유지 가능
LAST_PRED_DIR = REPORTS_DIR / "_last_prediction"
LAST_PRED_JSON = LAST_PRED_DIR / "last_prediction_meta.json"
LAST_PRED_CSV = LAST_PRED_DIR / "last_prediction_out.csv"

st.set_page_config(page_title="NGS MoA Classifier Dashboard", layout="wide")

st.title("🧬 NGS-MoA-Classifier")
st.caption("LOOCV / Permutation test / Selected genes / Enrichment / Figures 확인용")

# ---------------------------
# 사이드바
# ---------------------------
st.sidebar.header("메뉴")
page = st.sidebar.radio(
    "보고 싶은 화면 선택",
    ["Overview", "Data", "Predict", "Model Metrics", "Model Comparison", "Selected Genes", "Enrichment", "Figures"],
)

def file_exists(p: Path) -> bool:
    return p.exists() and p.is_file()

# ---------------------------
# ✅ (추가) Predict 결과를 Overview에 자동 반영하기 위한 세션 + 파일 저장
#  - session_state: 같은 세션/탭에서 가장 빠르고 간단
#  - file 저장: rerun/탭 전환/가끔 세션 꼬임에도 안전하게 복구
# ---------------------------
def save_last_prediction(payload: dict):
    """
    Predict 페이지에서 생성된 결과를 st.session_state에 저장해서
    Overview에서 '최근 업로드 예측결과'로 자동 표시되게 함

    ✅ (추가) 동시에 reports/_last_prediction/ 에도 저장해서
    세션이 꼬이거나 초기화돼도 Overview에서 복구 가능하게 함
    """
    st.session_state["last_prediction"] = payload

    try:
        LAST_PRED_DIR.mkdir(parents=True, exist_ok=True)

        # DataFrame은 메타 JSON에 직접 못 담으니 CSV로 분리 저장
        out_df = payload.get("out_df")
        meta = dict(payload)
        meta["out_df"] = None  # JSON에는 넣지 않음

        with open(LAST_PRED_JSON, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if isinstance(out_df, pd.DataFrame):
            out_df.to_csv(LAST_PRED_CSV, index=False, encoding="utf-8-sig")
    except Exception:
        # 파일 저장 실패해도 세션 저장은 됐으므로 조용히 넘어감
        pass

def get_last_prediction():
    """
    저장된 최근 예측 결과를 꺼냄 (없으면 None)

    우선순위:
    1) st.session_state["last_prediction"]
    2) reports/_last_prediction/ 저장 파일(JSON + CSV)에서 복구
    """
    last = st.session_state.get("last_prediction")
    if last is not None:
        return last

    # 세션에 없으면 파일에서 복구 시도
    if file_exists(LAST_PRED_JSON):
        try:
            meta = json.loads(LAST_PRED_JSON.read_text(encoding="utf-8"))
            if file_exists(LAST_PRED_CSV):
                try:
                    meta["out_df"] = pd.read_csv(LAST_PRED_CSV)
                except Exception:
                    meta["out_df"] = None
            st.session_state["last_prediction"] = meta
            return meta
        except Exception:
            return None

    return None

def clear_last_prediction():
    st.session_state.pop("last_prediction", None)
    try:
        if file_exists(LAST_PRED_JSON):
            LAST_PRED_JSON.unlink()
        if file_exists(LAST_PRED_CSV):
            LAST_PRED_CSV.unlink()
    except Exception:
        pass

# ---------------------------
# ✅ (추가) 모델 로드 관련 캐시
#  - Streamlit은 입력(체크박스/업로드 등) 바뀔 때마다 rerun되므로
#    모델 로드를 캐싱해서 '렉' 느낌을 줄이는 게 좋음
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: Path):
    import joblib
    return joblib.load(model_path)

# ---------------------------
# Overview
# ---------------------------
if page == "Overview":
    st.subheader("프로젝트 경로")
    st.write({
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "PROCESSED_DIR": str(PROCESSED_DIR),
        "MODELS_DIR": str(MODELS_DIR),
        "REPORTS_DIR": str(REPORTS_DIR),
        "FIGURES_DIR": str(FIGURES_DIR),
    })

    metrics_path = REPORTS_DIR / "metrics.json"
    if file_exists(metrics_path):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Samples", metrics.get("n_samples"))
        c2.metric("Genes (before)", metrics.get("n_genes_before"))
        c3.metric("Genes (after filter)", metrics.get("n_genes_after_filter"))
        c4.metric("Selected genes", metrics.get("n_selected_genes"))
        st.success("metrics.json 로드 완료")
    else:
        st.warning("reports/metrics.json 이 없어요. 먼저 train.py 실행해서 생성해줘야 함.")

    # ---------------------------
    # ✅ (추가) 최근 업로드 예측결과(자동 반영)
    # ---------------------------
    st.divider()
    st.subheader("최근 업로드 예측결과 (Predict → 자동 반영)")

    last = get_last_prediction()
    if last is None:
        st.info("아직 Predict에서 업로드/예측한 기록이 없어요. Predict에서 파일 업로드 후 자동으로 여기에 표시됩니다.")
    else:
        oc1, oc2, oc3, oc4 = st.columns(4)
        oc1.metric("업로드 파일명", last.get("file_name", "-"))
        oc2.metric("Samples(업로드)", last.get("n_samples", 0))
        oc3.metric("Treated(1)", last.get("n_treated", 0))
        oc4.metric("Control(0)", last.get("n_control", 0))

        st.caption(
            f"Genes(업로드): {last.get('n_genes_upload', '-')}, "
            f"Expected genes: {last.get('n_expected', '-')}, "
            f"Missing: {last.get('n_missing', '-')}, Extra: {last.get('n_extra', '-')}"
        )

        out_df = last.get("out_df")
        if isinstance(out_df, pd.DataFrame):
            st.dataframe(out_df, use_container_width=True)

            st.download_button(
                "최근 예측 결과 CSV 다운로드",
                data=out_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="predict_results_last.csv",
                mime="text/csv",
            )

        if st.button("최근 예측결과 지우기"):
            clear_last_prediction()
            st.rerun()

# ---------------------------
# Data
# ---------------------------
elif page == "Data":
    st.subheader("Processed Data 확인")

    x_path = PROCESSED_DIR / "X_gene_expression.csv"
    y_path = PROCESSED_DIR / "y_labels.csv"

    if file_exists(x_path):
        X = pd.read_csv(x_path)
        st.write("✅ X_gene_expression.csv")
        st.dataframe(X.head(20), use_container_width=True)
        st.caption(f"shape: {X.shape}")
    else:
        st.error("data/processed/X_gene_expression.csv 없음")

    if file_exists(y_path):
        y = pd.read_csv(y_path)
        st.write("✅ y_labels.csv")
        st.dataframe(y, use_container_width=True)
        st.caption(f"shape: {y.shape}")
    else:
        st.error("data/processed/y_labels.csv 없음")

# ---------------------------
# Predict
# ---------------------------
elif page == "Predict":
    st.subheader("Predict (업로드 → 자동 정렬/보정 → 예측)")

    import numpy as np

    MIN_NONZERO_FRAC = 0.34  # 학습과 동일
    model_path = MODELS_DIR / "voting_rf_lr.joblib"

    # ✅ (추가) 업로드 데이터가 다양한 형식일 수 있어서 옵션 제공
    with st.expander("업로드 옵션(필요할 때만)", expanded=False):
        only_readcount = st.checkbox("샘플명에 'Read_Count'가 포함된 행만 사용", value=True)
        drop_allzero_samples = st.checkbox("합이 0인 샘플(행) 제거", value=True)

        # ✅ (추가) 미리보기 표시 행 수(너무 크게 잡으면 렉 체감)
        preview_rows = st.slider("미리보기 행 수", 5, 80, 30, step=5)

    def wide_to_samplesxgenes(df: pd.DataFrame) -> pd.DataFrame:
        # gene 컬럼 찾기 (gene / Gene_Symbol 허용)
        gene_col = None
        for cand in ["gene", "Gene_Symbol", "gene_symbol"]:
            if cand in df.columns:
                gene_col = cand
                break
        if gene_col is None:
            raise ValueError("gene 컬럼을 못 찾았어요. (gene 또는 Gene_Symbol 필요)")

        sample_cols = [c for c in df.columns if c != gene_col]
        if len(sample_cols) == 0:
            raise ValueError("샘플 컬럼이 없어요. gene 컬럼 외에 숫자 샘플 컬럼이 있어야 함")

        tmp = df[[gene_col] + sample_cols].copy()
        for c in sample_cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0)

        # ✅ (수정) gene 중복 처리: 같은 gene가 여러 번 나오면 하나로 합치기(기본=sum)
        # - Expression_Profile 같은 파일에서 gene이 중복으로 존재하는 경우가 많아서
        #   그대로 columns로 쓰면 "Duplicate column names" 에러가 발생함
        tmp[gene_col] = tmp[gene_col].astype(str)

        tmp_agg = (
            tmp
            .groupby(gene_col, as_index=False)
            .sum(numeric_only=True)
        )

        genes = tmp_agg[gene_col].tolist()
        X = tmp_agg.drop(columns=[gene_col]).T  # samples x genes
        X.columns = genes
        X.index.name = "sample"

        # ✅ (추가) 업로드 원본에 annotation/메타데이터 행이 섞여 들어오는 경우 자동 제거
        # 예: sample 축에 Gene_ID / Transcript_ID / Description / gene_biotype / Protein_ID 같은 행이 들어옴
        # -> 이런 "샘플"은 실제 샘플이 아니므로 제거
        meta_like = {
            "Gene_ID", "gene_id",
            "Transcript_ID", "transcript_id",
            "Description", "description",
            "gene_biotype", "Gene_biotype",
            "Protein_ID", "protein_id",
            "Length", "length",
            "Chr", "chr", "Chromosome", "chromosome",
        }
        drop_idx = [i for i in X.index.astype(str) if i in meta_like]
        if drop_idx:
            X = X.drop(index=drop_idx, errors="ignore")

        # ✅ (추가) 전부 0인 샘플(행)은 제거 (메타데이터/빈 행이 남는 경우 방지)
        if drop_allzero_samples:
            X = X.loc[X.sum(axis=1) > 0]

        # ✅ (추가) 실제 샘플 행만 유지 (Read_Count 기준) - 필요 시 옵션으로 끌 수 있게 함
        if only_readcount:
            X = X.loc[X.index.astype(str).str.contains("Read_Count", na=False)]

        return X

    def log2_transform(X: pd.DataFrame) -> pd.DataFrame:
        return np.log2(X + 1.0)

    def get_expected_genes_from_model(model):
        # sklearn이 DataFrame으로 fit되면 feature_names_in_이 남아있음
        for step in ["scaler", "select", "clf"]:
            if hasattr(model, "named_steps") and step in model.named_steps:
                obj = model.named_steps[step]
                if hasattr(obj, "feature_names_in_"):
                    return list(obj.feature_names_in_)
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        return None

    @st.cache_data(show_spinner=False)
    def expected_genes_fallback_from_training_processed() -> list[str]:
        """
        feature_names_in_이 없을 때 대비:
        data/processed/X_gene_expression.csv를 학습과 동일하게 log2 + low-expression filter로 처리해
        학습 당시 X2 컬럼 리스트(=모델 입력 컬럼)를 복원
        """
        x_path = PROCESSED_DIR / "X_gene_expression.csv"
        if not (x_path.exists() and x_path.is_file()):
            raise FileNotFoundError("feature_names_in_이 없고, data/processed/X_gene_expression.csv도 없습니다.")

        raw = pd.read_csv(x_path)

        gene_col = "gene" if "gene" in raw.columns else ("Gene_Symbol" if "Gene_Symbol" in raw.columns else None)
        if gene_col is None:
            raise ValueError("X_gene_expression.csv에서 gene 컬럼을 못 찾았어요.")

        sample_cols = [c for c in raw.columns if c != gene_col]
        tmp = raw[[gene_col] + sample_cols].copy()
        for c in sample_cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0)

        genes = tmp[gene_col].astype(str).tolist()
        X = tmp.drop(columns=[gene_col]).T
        X.columns = genes

        # 학습과 동일: log2 -> low expression filter(genes 기준)
        X2 = log2_transform(X)
        nonzero_frac = (X2 > 0).sum(axis=0) / X2.shape[0]
        keep = nonzero_frac >= MIN_NONZERO_FRAC
        X2 = X2.loc[:, keep]

        return list(X2.columns)

    def align_to_expected(X_new: pd.DataFrame, expected_genes: list[str]) -> tuple[pd.DataFrame, dict]:
        """
        - expected_genes 순서로 컬럼 정렬
        - 누락된 유전자는 0으로 추가
        - extra 유전자는 제거
        """
        new_genes = set(X_new.columns.astype(str))

        missing = [g for g in expected_genes if g not in new_genes]
        extra = [g for g in X_new.columns if str(g) not in set(expected_genes)]

        X_aligned = X_new.copy()
        # 누락 유전자 0으로 추가
        for g in missing:
            X_aligned[g] = 0.0

        # expected만 남기고 순서 고정
        X_aligned = X_aligned.loc[:, expected_genes]

        info = {
            "n_expected": len(expected_genes),
            "n_input_genes": X_new.shape[1],
            "n_missing": len(missing),
            "n_extra": len(extra),
            "missing_rate": (len(missing) / max(1, len(expected_genes))),
        }
        return X_aligned, info

    # --------------------------
    # 1) 모델 로드 (캐시 적용)
    # --------------------------
    if not (model_path.exists() and model_path.is_file()):
        st.error("models/voting_rf_lr.joblib 없음 (먼저 train.py 실행해서 모델 저장 필요)")
        st.stop()

    model = load_model_cached(model_path)
    st.success("✅ 모델 로드 완료")

    # --------------------------
    # 2) 모델이 기대하는 유전자 컬럼 확보
    # --------------------------
    expected_genes = get_expected_genes_from_model(model)
    if expected_genes is None:
        st.warning("모델에 feature_names_in_이 없어서 processed 데이터로 기대 컬럼을 복원합니다.")
        try:
            expected_genes = expected_genes_fallback_from_training_processed()
        except Exception as e:
            st.error("기대 유전자 컬럼을 복원하지 못했어요.")
            st.exception(e)
            st.stop()

    st.caption(f"모델 입력 유전자 수(expected): {len(expected_genes)}")

    # --------------------------
    # 3) 업로드 + 실행을 Form으로 묶기 (✅ 핵심: Overview 자동 반영이 더 안정적)
    #   - 업로드만 했을 때는 저장/예측이 안 됨
    #   - '예측 실행'을 눌렀던 그 run에서 예측→save_last_prediction까지 한 번에 완료
    # --------------------------
    with st.form("predict_form", clear_on_submit=False):
        up = st.file_uploader(
            "예측할 데이터 업로드 (xlsx 또는 csv)",
            type=["xlsx", "csv"],
            key="predict_uploader",  # ✅ key 고정(세션 안정)
        )
        st.caption("A안 형식: 첫 컬럼 gene(또는 Gene_Symbol), 나머지는 샘플 컬럼(원시 Read Count 등 숫자)")
        run_pred = st.form_submit_button("예측 실행")

    if up is None:
        st.info("파일을 업로드한 뒤, 아래 '예측 실행' 버튼을 누르면 예측이 실행됩니다.")
        st.stop()

    if not run_pred:
        st.info("파일은 업로드됐어요. **'예측 실행'**을 누르면 전처리/정렬/예측이 진행됩니다.")
        st.stop()

    # --------------------------
    # 4) 파일 읽기 → samples x genes
    # --------------------------
    try:
        with st.spinner("업로드 파일을 읽고 전처리 중..."):
            # ✅ rerun에도 덜 꼬이도록 BytesIO로 읽기
            bio = BytesIO(up.getvalue())
            if up.name.lower().endswith(".xlsx"):
                df_up = pd.read_excel(bio)
            else:
                df_up = pd.read_csv(bio)

            X_new_raw = wide_to_samplesxgenes(df_up)
    except Exception as e:
        st.error("업로드 파일 파싱 실패")
        st.exception(e)
        st.stop()

    # ✅ (추가) 업로드/전처리 요약(Overview 스타일)
    st.subheader("업로드 데이터 요약")
    uc1, uc2, uc3, uc4 = st.columns(4)
    uc1.metric("Samples (업로드)", int(X_new_raw.shape[0]))
    uc2.metric("Genes (업로드)", int(X_new_raw.shape[1]))
    uc3.metric("Expected genes", int(len(expected_genes)))
    uc4.metric("업로드 파일명", up.name)

    with st.expander("업로드 원본(가공 전) 미리보기", expanded=True):
        st.dataframe(X_new_raw.head(preview_rows), use_container_width=True)
        st.caption(f"shape (samples x genes): {X_new_raw.shape}")

        # ✅ (추가) 샘플 목록 보여주기(미리보기에서 5개만 보이는 것처럼 느껴질 때 확인용)
        st.write("샘플 목록")
        st.write(list(X_new_raw.index.astype(str)))

    # --------------------------
    # 5) expected 컬럼에 맞춰 정렬/보정
    # --------------------------
    with st.spinner("모델 입력 유전자에 맞춰 정렬/보정 중..."):
        X_aligned, info = align_to_expected(X_new_raw, expected_genes)

    st.subheader("정렬/보정 결과 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected genes", info["n_expected"])
    c2.metric("Input genes", info["n_input_genes"])
    c3.metric("Missing genes", info["n_missing"])
    c4.metric("Extra genes", info["n_extra"])

    if info["missing_rate"] > 0.3:
        st.warning(f"누락 유전자가 많아요(≈{info['missing_rate']*100:.1f}%). 예측 신뢰도가 떨어질 수 있습니다.")

    with st.expander("정렬/보정된 입력(X_aligned) 미리보기", expanded=False):
        st.dataframe(X_aligned.head(10), use_container_width=True)
        st.caption(f"shape: {X_aligned.shape}")

    # --------------------------
    # 6) 학습과 동일하게 log2(+1) 적용
    # --------------------------
    X_input = log2_transform(X_aligned)

    # --------------------------
    # 7) 예측
    # --------------------------
    try:
        with st.spinner("예측 수행 중..."):
            proba = model.predict_proba(X_input)[:, 1]
            pred = (proba >= 0.5).astype(int)
    except Exception as e:
        st.error("예측 실패: 모델 입력 컬럼/형식 불일치 가능성이 큼")
        st.exception(e)
        st.stop()

    out = pd.DataFrame({
        "sample": X_input.index.astype(str),
        "pred": pred,
        "proba_treated": proba
    }).sort_values("proba_treated", ascending=False)

    # ✅ (추가) Predict 결과를 Overview에 자동 반영(세션 + 파일 저장)
    save_last_prediction({
        "file_name": up.name,
        "n_samples": int(X_input.shape[0]),
        "n_genes_upload": int(X_new_raw.shape[1]),
        "n_expected": int(info["n_expected"]),
        "n_missing": int(info["n_missing"]),
        "n_extra": int(info["n_extra"]),
        "n_treated": int((pred == 1).sum()),
        "n_control": int((pred == 0).sum()),
        "out_df": out,  # 샘플 수가 많지 않으면 그대로 저장 OK (파일로도 저장됨)
    })

    st.success("✅ 예측 완료")

    st.subheader("예측 결과")
    st.dataframe(out, use_container_width=True)

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Treated(1) 예측", int((pred == 1).sum()))
    pc2.metric("Control(0) 예측", int((pred == 0).sum()))
    pc3.metric("평균 proba(1)", float(np.mean(proba)) if len(proba) else 0.0)

    st.download_button(
        "예측 결과 CSV 다운로드",
        data=out.to_csv(index=False).encode("utf-8-sig"),
        file_name="predict_results.csv",
        mime="text/csv",
    )

    # ✅ (추가) Predict에서 Overview/대시보드 요약까지 같이 보여주기
    with st.expander("대시보드 요약(학습 metrics.json) 함께 보기", expanded=False):
        metrics_path = REPORTS_DIR / "metrics.json"
        if file_exists(metrics_path):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Train Samples", metrics.get("n_samples"))
            cc2.metric("Train Genes(before)", metrics.get("n_genes_before"))
            cc3.metric("Train Genes(after)", metrics.get("n_genes_after_filter"))
            cc4.metric("Selected genes", metrics.get("n_selected_genes"))
            st.caption("※ 이 값들은 학습 결과(train.py) 기준이며, 업로드 데이터 요약과는 별개입니다.")
        else:
            st.warning("reports/metrics.json 이 없어요. 먼저 train.py 실행해서 생성해줘야 함.")

    # ✅ (추가) 결과를 더 이어서 보고 싶을 때를 위한 힌트
    st.caption("다음으로: 왼쪽 메뉴에서 Selected Genes / Enrichment / Figures로 이동하면 학습 결과 리포트를 이어서 확인할 수 있어요.")
    st.caption("또는 Overview로 이동하면 방금 예측 결과가 '최근 업로드 예측결과' 영역에 자동 반영됩니다.")

# ---------------------------
# Model Metrics
# ---------------------------
elif page == "Model Metrics":
    st.subheader("LOOCV Metrics (train.py 결과)")

    metrics_path = REPORTS_DIR / "metrics.json"
    if file_exists(metrics_path):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.json(metrics)

        if "confusion_matrix" in metrics:
            st.write("Confusion Matrix")
            cm = pd.DataFrame(metrics["confusion_matrix"], index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
            st.dataframe(cm, use_container_width=True)
    else:
        st.error("reports/metrics.json 없음")

# ---------------------------
# Model Comparison
# ---------------------------
elif page == "Model Comparison":
    st.subheader("Model Comparison + Permutation Test")

    comp_path = REPORTS_DIR / "model_comparison_permutation.csv"
    if file_exists(comp_path):
        df = pd.read_csv(comp_path)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("reports/model_comparison_permutation.csv 없음 (train_compare_permutation.py 실행 필요)")

    perm_fig = FIGURES_DIR / "permutation_hist_logistic_l2.png"
    if file_exists(perm_fig):
        st.image(str(perm_fig), caption="Permutation test histogram (Logistic_L2)")

# ---------------------------
# Selected Genes
# ---------------------------
elif page == "Selected Genes":
    st.subheader("Selected Genes (Top)")

    sel_path = REPORTS_DIR / "selected_genes_top.csv"
    if file_exists(sel_path):
        df = pd.read_csv(sel_path)
        topn = st.slider("Top N", 10, 300, 50, step=10)
        st.dataframe(df.head(topn), use_container_width=True)
    else:
        st.error("reports/selected_genes_top.csv 없음")

    akt_path = REPORTS_DIR / "akt_emt_candidate_genes.csv"
    if file_exists(akt_path):
        st.write("AKT/EMT 후보 유전자")
        akt = pd.read_csv(akt_path)
        st.dataframe(akt.head(100), use_container_width=True)

# ---------------------------
# Enrichment
# ---------------------------
elif page == "Enrichment":
    st.subheader("Enrichment 결과")

    top30 = REPORTS_DIR / "enrichment_results_top30.csv"
    allr = REPORTS_DIR / "enrichment_results_all.csv"
    top5 = REPORTS_DIR / "table3_enrichment_top5.csv"

    if file_exists(top5):
        st.write("Top5 요약(table3)")
        st.dataframe(pd.read_csv(top5), use_container_width=True)

    if file_exists(top30):
        st.write("Top30")
        st.dataframe(pd.read_csv(top30), use_container_width=True)

    if file_exists(allr):
        with st.expander("전체 Enrichment 결과 보기 (all)"):
            st.dataframe(pd.read_csv(allr).head(200), use_container_width=True)

# ---------------------------
# Figures
# ---------------------------
elif page == "Figures":
    st.subheader("Figures")

    # 네가 만든 그림 파일들 기준
    candidates = [
        FIGURES_DIR / "heatmap_top30.png",
        FIGURES_DIR / "pca_samples.png",
        FIGURES_DIR / "pca_topgenes.png",
        FIGURES_DIR / "sample_correlation.png",
        FIGURES_DIR / "volcano.png",
        FIGURES_DIR / "permutation_hist_logistic_l2.png",
        FIGURES_DIR / "topgenes_boxplot.png",
    ]

    found = [p for p in candidates if file_exists(p)]
    if not found:
        st.warning("figures 폴더에 표시할 이미지가 없어요. train.py 실행해서 figures 생성해줘.")
    else:
        for p in found:
            st.image(str(p), caption=p.name)
