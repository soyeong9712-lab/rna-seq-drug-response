import re
import pandas as pd
from src.utils.paths import RAW_DIR, PROCESSED_DIR
from src.utils.logger import get_logger

# 데이터 로딩/라벨링/저장

logger = get_logger()

def infer_label(sample_name: str) -> int:
    """
    처리군(12b) = 1, 대조군(Control) = 0
    샘플명: SW48-48h-12b1_Read_Count, SW48-48h-C1_Read_Count ...
    """
    s = str(sample_name).lower()

    # 뒤에 붙는 표현형 접미사 제거 (_Read_Count, _TPM, _FPKM 등)
    base = re.split(r"_(read_count|tpm|fpkm).*", s)[0]

    # treated
    if "12b" in base:
        return 1

    # control: -C1, -C2, -C3 (뒤에 뭐가 붙어도 OK)
    if re.search(r"-c\d+$", base):
        return 0

    raise ValueError(f"라벨 추론 실패: sample={sample_name}")


def main():
    input_path = RAW_DIR / "Expression_Profile.GRCh38.gene.xlsx"
    if not input_path.exists():
        raise FileNotFoundError(f"데이터 파일이 없음: {input_path}")

    logger.info(f"Load: {input_path}")
    df = pd.read_excel(input_path)

    # 보통 첫 컬럼이 gene id/ gene symbol 등의 정보이고, 나머지가 샘플들
    # 샘플 컬럼 추출(숫자형 expression 컬럼들)
    # 안전하게: object 제외하고 숫자형 컬럼만 샘플로 잡으면 안될 수 있어,
    # 여기선 '첫 1~3개 메타컬럼' 가능성을 고려해 샘플명을 기준으로 추출
    cols = list(df.columns)

    # 샘플 컬럼 후보: "SW48" 포함한 컬럼들
    sample_cols = [c for c in cols if isinstance(c, str) and ("SW48" in c) and c.endswith("_Read_Count")]

    if len(sample_cols) == 0:
        raise ValueError(
            "샘플 컬럼을 찾지 못했어요. 컬럼명에 'SW48'이 없다면 make_dataset.py에서 sample_cols 규칙을 바꿔야 합니다."
        )

    # 유전자 식별 컬럼(메타) 후보: sample_cols 제외한 앞쪽 컬럼들
    meta_cols = [c for c in cols if c not in sample_cols]

# gene symbol 컬럼 사용 (MoA 해석 품질 향상)
    gene_id_col = "Gene_Symbol"
    if gene_id_col not in df.columns:
        raise ValueError(f"{gene_id_col} 컬럼이 엑셀에 없습니다.")

    logger.info(f"Gene symbol column used: {gene_id_col}")
    df = df[[gene_id_col] + sample_cols].copy()
    df.rename(columns={gene_id_col: "gene"}, inplace=True)


    # long -> wide 정리: (여기선 이미 wide 형태: gene_id rows, samples columns)
    # label 테이블 생성
    labels = pd.DataFrame({
        "sample": sample_cols,
        "y": [infer_label(s) for s in sample_cols]
    })

    # 저장
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_x = PROCESSED_DIR / "X_gene_expression.csv"
    out_y = PROCESSED_DIR / "y_labels.csv"

    df.to_csv(out_x, index=False)
    labels.to_csv(out_y, index=False)

    logger.info(f"Saved X: {out_x}")
    logger.info(f"Saved y: {out_y}")
    logger.info(f"Class counts:\n{labels['y'].value_counts()}")

if __name__ == "__main__":
    main()
