# src/models/predict_external.py
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# -----------------------------
# 0) DNN 모델 정의 (학습 때 구조와 동일해야 함)
# -----------------------------
class GeneExpressionDNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------
# 1) 프로젝트 루트 찾기
# -----------------------------
def _find_project_root() -> Path:
    current_path = Path(__file__).resolve()
    for p in current_path.parents:
        if p.name == "ngs-moa-classifier":
            return p
    return current_path.parents[2]


# -----------------------------
# 2) 학습/추론 공통 전처리 (log2(x+1))
#    ※ 주의: 외부가 이미 log2일 수도 있음. (필요시 아래 run_prediction에서 스위치 가능)
# -----------------------------
def log2_transform(df: pd.DataFrame) -> pd.DataFrame:
    return np.log2(df.astype(float) + 1.0)


# -----------------------------
# 3) GEO series_matrix 파서: expression table만 추출
# -----------------------------
def load_gse_series_matrix(series_path: Path) -> pd.DataFrame:
    """
    GEO series_matrix.txt에서 expression table만 뽑아
    rows(ID=probe/transcript_cluster_id 등) x samples DataFrame 반환
    """
    lines = series_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    begin_idx, end_idx = None, None
    for i, line in enumerate(lines):
        l = line.strip().lower()
        if l == "!series_matrix_table_begin":
            begin_idx = i + 1
        elif l == "!series_matrix_table_end":
            end_idx = i
            break

    if begin_idx is None or end_idx is None or begin_idx >= end_idx:
        raise ValueError("series_matrix_table_begin/end를 찾지 못했습니다. series_matrix 형식을 확인하세요.")

    table_lines = lines[begin_idx:end_idx]
    df = pd.read_csv(StringIO("\n".join(table_lines)), sep="\t")

    first_col = df.columns[0]
    df = df.set_index(first_col)

    # 숫자 변환
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


# -----------------------------
# 4) GPL6244 annotation 파싱: ID(probe) -> Gene Symbol
#    (네가 해결한 안정형 버전 유지)
# -----------------------------
def build_probe_to_symbol_from_gpl(gpl_path: Path) -> dict[str, str]:
    """
    GPL6244-17930.txt ('Download full table')를 읽어서
    ID -> gene symbol 매핑 딕셔너리 생성

    - comment='#' 로 메타 라인 제거
    - gene_assignment 컬럼에서 ' // SYMBOL // ' 패턴 우선 추출
    """
    # engine='c'가 가장 안정/빠름. (python 엔진 쓰면 low_memory 옵션 충돌 가능)
    df = pd.read_csv(
        gpl_path,
        sep="\t",
        comment="#",
        dtype=str,
        low_memory=False,   # c엔진에서는 OK
        engine="c",
    )

    if df.empty:
        raise ValueError("GPL 파일을 읽었지만 테이블이 비어있습니다. 파일이 정상인지 확인하세요.")

    df.columns = [str(c).strip() for c in df.columns]
    if "ID" not in df.columns:
        raise ValueError(f"GPL 파일에 'ID' 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()[:30]}")

    # symbol 소스 컬럼 찾기
    symbol_col_candidates = [
        "Gene Symbol", "Gene symbol", "gene_symbol", "GENE_SYMBOL",
        "gene_assignment", "GENE_ASSIGNMENT",
        "SYMBOL", "Symbol",
    ]
    symbol_col = None
    for c in symbol_col_candidates:
        if c in df.columns:
            symbol_col = c
            break
    if symbol_col is None:
        # assignment 비슷한 컬럼 fallback
        for c in df.columns:
            if "assignment" in c.lower():
                symbol_col = c
                break
    if symbol_col is None:
        raise ValueError("GPL 파일에서 gene symbol 컬럼을 찾지 못했습니다. 컬럼 구성을 확인하세요.")

    def extract_symbol(v: str) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            return None

        # 1) // 패턴 우선
        if "//" in s:
            parts = [p.strip() for p in s.split("//")]
            if len(parts) >= 2 and parts[1] and parts[1] not in ("---", "-"):
                sym = parts[1].split()[0].strip()
                if sym and sym.lower() not in ("na", "nan", "---"):
                    return sym

        # 2) 심볼처럼 생긴 토큰
        if re.match(r"^[A-Za-z0-9_.\-]+$", s):
            return s

        # 3) 마지막 fallback
        tokens = re.findall(r"[A-Z0-9][A-Z0-9_\-]{1,}", s)
        return tokens[0] if tokens else None

    id_series = df["ID"].astype(str).str.strip()
    sym_series = df[symbol_col].astype(str).map(extract_symbol)

    m = pd.DataFrame({"ID": id_series, "SYMBOL": sym_series}).dropna()
    m["ID"] = m["ID"].astype(str).str.strip()
    m["SYMBOL"] = m["SYMBOL"].astype(str).str.strip()
    m = m.drop_duplicates(subset=["ID"], keep="first")

    return dict(zip(m["ID"], m["SYMBOL"]))


def map_probe_to_symbol_and_collapse(df_ext_id_x_samples: pd.DataFrame, probe2sym: dict[str, str]) -> pd.DataFrame:
    """
    외부 데이터 index(ID=probe/transcript_cluster_id)를 Gene Symbol로 변환하고,
    같은 symbol로 여러 probe가 매핑되면 합(sum)으로 collapse.
    """
    idx = df_ext_id_x_samples.index.astype(str).str.strip()
    mapped = idx.map(lambda x: probe2sym.get(x, np.nan))

    df = df_ext_id_x_samples.copy()
    df.index = mapped

    before = df.shape[0]
    df = df.dropna(axis=0)
    after = df.shape[0]
    print(f"   - probe→symbol 매핑 성공: {after}/{before} rows")

    # 같은 심볼로 collapse
    df = df.groupby(df.index).sum()
    return df


# -----------------------------
# 5) 학습 데이터(X/y) 로드: (samples x genes) 형태 보장
# -----------------------------
def load_train_xy(feature_x_path: Path, feature_y_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(feature_x_path, index_col=0)
    y_df = pd.read_csv(feature_y_path, index_col=0)

    if y_df.shape[1] == 1:
        y = y_df.iloc[:, 0]
    else:
        y = y_df["y"] if "y" in y_df.columns else y_df.iloc[:, 0]

    # ✅ X가 genes x samples로 저장됐으면 transpose (너 케이스가 이거)
    if X.shape[0] > X.shape[1]:
        X = X.T

    y = y.reindex(X.index)
    return X, y


# -----------------------------
# 6) state_dict key 자동 보정 로더
#    - "model." prefix가 있거나 없거나 자동 처리
# -----------------------------
def load_model_state_safely(model: nn.Module, model_path: Path) -> None:
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    keys = list(state.keys())
    has_model_prefix = any(k.startswith("model.") for k in keys)

    # 현재 model.state_dict() 키가 "model.0.weight" 형태인지 확인
    target_keys = list(model.state_dict().keys())
    target_has_model_prefix = any(k.startswith("model.") for k in target_keys)

    cleaned = {}
    for k, v in state.items():
        nk = k

        # 저장은 "0.weight"인데, 현재 모델은 "model.0.weight"를 기대하는 경우 → prefix 추가
        if (not has_model_prefix) and target_has_model_prefix:
            nk = "model." + nk

        # 저장은 "model.0.weight"인데, 현재 모델은 "0.weight"를 기대하는 경우 → prefix 제거
        if has_model_prefix and (not target_has_model_prefix) and nk.startswith("model."):
            nk = nk[len("model."):]

        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=True)


# -----------------------------
# 7) 실행
# -----------------------------
def run_prediction():
    project_root = _find_project_root()

    model_path = project_root / "models" / "dnn_model.pth"
    feature_x_path = project_root / "data" / "processed" / "X_gene_expression.csv"
    feature_y_path = project_root / "data" / "processed" / "y_labels.csv"

    ext_path = project_root / "data" / "raw" / "external" / "GSE100928_series_matrix.txt"
    gpl_path = project_root / "data" / "raw" / "external" / "GPL6244-17930.txt"

    print("--- 🔍 Step 1: 파일 존재 확인 ---")
    files = {
        "모델(dnn_model.pth)": model_path,
        "학습 X": feature_x_path,
        "학습 y": feature_y_path,
        "외부 series_matrix": ext_path,
        "GPL annotation": gpl_path,
    }
    for name, path in files.items():
        print(f"{name}: {'✅ 있음' if path.exists() else '❌ 없음'} -> {path}")

    must = [model_path, feature_x_path, feature_y_path, ext_path, gpl_path]
    if not all(p.exists() for p in must):
        print("🛑 중단: 필수 파일이 부족합니다.")
        return

    # ---------------------------------------
    # Step 2: 학습 feature 로드 + 학습 기준 mean/std 계산 (log-space 기준)
    # ---------------------------------------
    print("\n--- ⚙️ Step 2: 학습 feature(유전자) 불러오기 ---")
    X_train, y_train = load_train_xy(feature_x_path, feature_y_path)
    print(f"학습 X shape: {X_train.shape} (samples x genes)")
    print(f"학습 y values: {y_train.values}")

    X_train_log = log2_transform(X_train)
    train_genes = list(X_train_log.columns)
    input_dim = len(train_genes)
    print(f"학습 유전자 수(input_dim): {input_dim}")

    # 학습에서 사용한 정규화 파라미터
    X_mean = X_train_log.mean(axis=0)          # Series (genes)
    X_std = X_train_log.std(axis=0) + 1e-8     # Series (genes)

    # ---------------------------------------
    # Step 3: 외부 데이터 로드 (series_matrix)
    # ---------------------------------------
    print("\n--- 🧬 Step 3: 외부 데이터 로드 (series_matrix) ---")
    df_ext = load_gse_series_matrix(ext_path)  # rows(probe) x samples
    print(f"외부 원본 shape (rows x samples): {df_ext.shape}")

    # ---------------------------------------
    # Step 3-1: GPL 기반 probe -> symbol 매핑
    # ---------------------------------------
    print("\n--- 🔁 Step 3-1: GPL 기반 probe→symbol 매핑 ---")
    probe2sym = build_probe_to_symbol_from_gpl(gpl_path)
    print(f"probe2sym 크기: {len(probe2sym):,}")

    df_ext_sym = map_probe_to_symbol_and_collapse(df_ext, probe2sym)  # rows(symbol) x samples
    print(f"외부 매핑 후 shape (genes(symbol) x samples): {df_ext_sym.shape}")

    # ---------------------------------------
    # Step 4: 학습 유전자에 맞춰 매칭
    #   ✅ 중요 수정: 매칭 안 된 유전자는 0 채우지 말고 NaN으로 두었다가,
    #               log2 후 '학습 평균(X_mean)'으로 채워서 정규화 후 0이 되게 만들기
    # ---------------------------------------
    print("\n--- 🧩 Step 4: 학습 유전자에 맞춰 매칭(누락 유전자 처리 개선) ---")
    X_ext = df_ext_sym.T  # samples x genes(symbol)
    
    # ✅ [추가] 메모리/속도 개선: float64 → float32
    X_ext = X_ext.astype(np.float32)
    
    before_genes = X_ext.shape[1]

    # 1) 학습 유전자에 맞춰 reindex (누락은 NaN 유지)
    X_ext = X_ext.reindex(columns=train_genes)

    # 2) log2 적용 (외부 데이터가 이미 log2일 수도 있음)
    #    필요하면 아래를 False로 바꾸고 log2 스킵 가능
    APPLY_LOG2 = True
    if APPLY_LOG2:
        X_ext_log = log2_transform(X_ext)
    else:
        X_ext_log = X_ext.astype(float)

    # 3) ✅ 누락 유전자는 "학습 평균(log-space)"로 채우기
    #    - 이렇게 하면 정규화 후 (mean-mean)/std = 0 이 되어 편향이 줄어듦
    X_ext_log = X_ext_log.fillna(X_mean)

    # 매칭률 계산(원본 값 기준이 아니라, '채우기 전' non-null 기준도 같이 보고 싶으면 별도 계산 가능)
    nonzero_per_feature = (X_ext_log != 0).sum(axis=0)
    matched_features = int((X_ext.notna().sum(axis=0) > 0).sum())  # 실제로 외부에서 관측된 유전자 수
    match_rate = matched_features / len(train_genes)

    print(f"외부 샘플 수: {X_ext_log.shape[0]}")
    print(f"매칭 전 외부 feature 수: {before_genes}")
    print(f"매칭 후 feature 수(=학습 유전자 수): {X_ext_log.shape[1]}")
    print(f"✅ 외부에서 실제 관측된 feature 수: {matched_features}/{len(train_genes)} (match_rate={match_rate:.3f})")
    print("   - reindex 완료. now computing match stats...")

    # 4) 정규화(학습 mean/std 기준)
    X_ext_norm = (X_ext_log - X_mean) / X_std

    # 5) 극단값 클리핑(옵션): logits 폭주/언더플로우 완화
    CLIP_Z = 8.0
    X_ext_norm = X_ext_norm.clip(lower=-CLIP_Z, upper=CLIP_Z)

    # NaN/Inf 방어
    X_ext_norm = X_ext_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ---------------------------------------
    # Step 5: 모델 로드 & 예측
    # ---------------------------------------
    print("\n--- 🚀 Step 5: 모델 로드 & 예측 ---")
    
    # ✅ [추가] 누락 유전자(NaN)는 학습 평균으로 채우기 (0으로 채우는 것보다 훨씬 안전)
    # - 외부 플랫폼에서 측정 안 된 유전자를 0으로 두면 log2(0+1)=0이 되어
    #   학습분포와 괴리 커지고 예측이 한쪽으로 쏠릴 수 있음
    X_ext = X_ext.fillna(X_mean.astype(np.float32))
    
    model = GeneExpressionDNN(input_dim=input_dim)
    load_model_state_safely(model, model_path)
    model.eval()
    print("✅ 모델 state_dict 로드 성공")

    X_tensor = torch.tensor(X_ext_norm.values.astype(np.float32))
    with torch.no_grad():
        probs = model(X_tensor).numpy().flatten()

    preds = (probs >= 0.5).astype(int)

    # ✅ 디버깅: 확률이 0만 나오는지 체크
    print(f"proba stats: min={probs.min():.6g}, max={probs.max():.6g}, mean={probs.mean():.6g}")

    results = pd.DataFrame(
        {"sample": X_ext_norm.index.astype(str), "proba_treated": probs.astype(float), "pred": preds.astype(int)}
    )

    print("✅ 예측 완료 (상위 10개 미리보기)")
    print(results.head(10).to_string(index=False))

    # ---------------------------------------
    # Step 6: 저장
    # ---------------------------------------
    print("\n--- 💾 Step 6: 결과 저장 ---")
    processed_save = project_root / "data" / "processed" / "external_test_results.csv"
    processed_save.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(processed_save, index=False, encoding="utf-8-sig")
    print(f"✅ 저장 성공: {processed_save}")

    last_dir = project_root / "reports" / "_last_prediction"
    last_dir.mkdir(parents=True, exist_ok=True)

    out_csv = last_dir / "last_prediction_out.csv"
    results.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 저장 성공: {out_csv}")

    meta = {
        "model": "GeneExpressionDNN",
        "model_path": str(model_path),
        "external_path": str(ext_path),
        "gpl_path": str(gpl_path),
        "APPLY_LOG2": APPLY_LOG2,
        "CLIP_Z": CLIP_Z,
        "train_X_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "n_external_samples": int(results.shape[0]),
        "n_features(input_dim)": int(input_dim),
        "matched_features_observed": int(matched_features),
        "match_rate_observed": float(match_rate),
        "n_pred_1": int((results["pred"] == 1).sum()),
        "n_pred_0": int((results["pred"] == 0).sum()),
        "proba_min": float(np.min(probs)),
        "proba_max": float(np.max(probs)),
        "proba_mean": float(np.mean(probs)),
    }

    meta_path = last_dir / "last_prediction_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✅ 저장 성공: {meta_path}")


if __name__ == "__main__":
    run_prediction()
