import numpy as np
import pandas as pd

# 전처리/피처선택 로직

def log2_transform(X: pd.DataFrame) -> pd.DataFrame:
    # expression은 0이 있을 수 있으니 +1
    return np.log2(X + 1.0)

def filter_low_expression(X: pd.DataFrame, min_nonzero_frac: float = 0.34) -> pd.DataFrame:
    """
    n=6 샘플이라 min_nonzero_frac=0.34면 최소 2개 샘플 이상에서 0이 아닌 유전자만 유지.
    (너무 빡세게 자르면 정보가 사라질 수 있어서 완만하게 시작)
    """
    nonzero_frac = (X > 0).sum(axis=1) / X.shape[1]
    keep = nonzero_frac >= min_nonzero_frac
    return X.loc[keep]

def load_processed(processed_x_path: str, processed_y_path: str):
    Xdf = pd.read_csv(processed_x_path)
    ydf = pd.read_csv(processed_y_path)

    # Xdf: gene_id(or gene) + sample columns
    gene_col = "gene_id" if "gene_id" in Xdf.columns else "gene"
    gene_ids = Xdf[gene_col].astype(str)

    # ✅ 여기만 바뀜
    X = Xdf.drop(columns=[gene_col])

    # shape: genes x samples  -> 모델은 samples x genes 형태가 필요
    X = X.T
    X.columns = gene_ids
    X.index.name = "sample"

    y = ydf.set_index("sample").loc[X.index, "y"].astype(int)
    return X, y

