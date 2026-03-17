import re
import pandas as pd
from gprofiler import GProfiler

from src.utils.paths import REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger()

def clean_genes(gene_series: pd.Series) -> list:
    genes = gene_series.astype(str).str.strip()

    # 결측/빈 값 제거
    genes = genes[genes.str.lower().ne("nan")]
    genes = genes[genes.ne("")]

    # 1) LOC 계열 제거 (미확정/예측 유전자 비중이 높아 매핑률 낮음)
    genes = genes[~genes.str.startswith("LOC")]

    # 2) miRNA 제거 (원하면 주석 처리)
    genes = genes[~genes.str.startswith("MIR")]

    # 3) Pseudogene(유사유전자) 많이 포함되면 enrichment가 빈약해질 수 있어 제거
    #    (원하면 주석 처리)
    genes = genes[~genes.str.contains(r"P\d+$", regex=True)]

    # 4) 심볼 형태 필터(너무 빡세면 주석 처리 가능)
    genes = genes[genes.str.match(r"^[A-Za-z0-9][A-Za-z0-9\-\._]*$")]

    # 중복 제거(순서 유지)
    cleaned = list(dict.fromkeys(genes.tolist()))
    return cleaned

def main():
    genes_path = REPORTS_DIR / "selected_genes_top.csv"
    if not genes_path.exists():
        raise FileNotFoundError(f"선택 유전자 파일 없음: {genes_path}")

    df = pd.read_csv(genes_path)

    top_n = 300
    raw = df["gene"].head(top_n)

    genes = clean_genes(raw)
    logger.info(f"Filtered genes: {len(genes)} (from top {top_n})")
    logger.info(f"Examples: {genes[:15]}")

    gp = GProfiler(return_dataframe=True)

    res = gp.profile(
        organism="hsapiens",
        query=genes,
        sources=["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC", "WP"]
    )

    out_all = REPORTS_DIR / "enrichment_results_all.csv"
    res.to_csv(out_all, index=False)
    logger.info(f"Saved: {out_all}")

    summary = res.sort_values("p_value").head(30)[
        ["source", "native", "name", "p_value", "intersection_size", "term_size"]
    ]
    out_top = REPORTS_DIR / "enrichment_results_top30.csv"
    summary.to_csv(out_top, index=False)
    logger.info(f"Saved: {out_top}")

if __name__ == "__main__":
    main()
