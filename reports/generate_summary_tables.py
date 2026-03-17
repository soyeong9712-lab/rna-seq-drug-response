"""
generate_summary_tables.py

수행내역서용 표 및 AKT/EMT 후보 유전자 추출 스크립트
- 표1: 전처리 단계별 유전자 수 변화
- 표2: LOOCV 성능 요약
- 표3: Enrichment 상위 결과
- AKT/EMT 관련 유전자 필터링
"""

import pandas as pd
from pathlib import Path

# ===========================
# 경로 설정 (프로젝트 루트 기준)
# ===========================
# 현재 스크립트 위치에서 프로젝트 루트 찾기
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "reports" else SCRIPT_DIR

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("📊 수행내역서용 표 및 후보 유전자 추출 시작")
print("="*80)
print(f"📂 프로젝트 루트: {PROJECT_ROOT}")
print(f"📂 Reports 폴더: {REPORTS_DIR}")
print("="*80 + "\n")

# ===========================
# 표 1: 전처리 단계별 유전자 수 변화
# ===========================
table1 = pd.DataFrame({
    "단계": [
        "Raw 데이터",
        "저발현 필터 후",
        "Feature Selection 후"
    ],
    "유전자 수": [
        46427,
        22607,
        300
    ],
    "비고": [
        "전체 유전자",
        "34% 기준 필터링 (6개 샘플 중 최소 2개 이상 발현)",
        "Logistic L2 기반 SelectFromModel"
    ]
})

table1_path = REPORTS_DIR / "table1_preprocessing_summary.csv"
table1.to_csv(table1_path, index=False, encoding="utf-8-sig")
print(f"✅ 표1 저장 완료: {table1_path}")
print(table1.to_string(index=False))
print("\n" + "="*80 + "\n")


# ===========================
# 표 2: LOOCV 성능 요약
# ===========================
table2 = pd.DataFrame({
    "지표": [
        "LOOCV Accuracy",
        "LOOCV ROC-AUC",
        "대조군 정확도 (0)",
        "처리군 정확도 (1)",
        "Confusion Matrix (0,0)",
        "Confusion Matrix (0,1)",
        "Confusion Matrix (1,0)",
        "Confusion Matrix (1,1)"
    ],
    "값": [
        "1.00",
        "1.00",
        "3/3",
        "3/3",
        "3",
        "0",
        "0",
        "3"
    ],
    "비고": [
        "6개 샘플 모두 정확 분류",
        "완벽한 분리",
        "대조군 3개 모두 정확",
        "처리군 3개 모두 정확",
        "True Negative",
        "False Positive",
        "False Negative",
        "True Positive"
    ]
})

table2_path = REPORTS_DIR / "table2_loocv_performance.csv"
table2.to_csv(table2_path, index=False, encoding="utf-8-sig")
print(f"✅ 표2 저장 완료: {table2_path}")
print(table2.to_string(index=False))
print("\n" + "="*80 + "\n")


# ===========================
# 표 3: Enrichment 상위 결과
# ===========================
table3 = pd.DataFrame({
    "Source": [
        "GO:CC",
        "GO:BP",
        "GO:BP",
        "KEGG",
        "GO:CC"
    ],
    "Term": [
        "Keratin filament",
        "Developmental process",
        "Anatomical structure development",
        "Arginine and proline metabolism",
        "Cell periphery"
    ],
    "p_value": [
        0.001289,
        0.007379,
        0.033828,
        0.038058,
        0.048296
    ],
    "Intersection_size": [
        7,
        68,
        62,
        4,
        65
    ],
    "Term_size": [
        93,
        6553,
        5997,
        48,
        6347
    ],
    "생물학적_의미": [
        "세포 골격 구조, EMT와 연관",
        "세포 분화 및 발달 프로그램",
        "구조적 발달 과정",
        "아미노산 대사 경로",
        "세포막/세포 상호작용, EMT 관련"
    ]
})

table3_path = REPORTS_DIR / "table3_enrichment_top5.csv"
table3.to_csv(table3_path, index=False, encoding="utf-8-sig")
print(f"✅ 표3 저장 완료: {table3_path}")
print(table3.to_string(index=False))
print("\n" + "="*80 + "\n")


# ===========================
# 2번: AKT/EMT 관련 유전자 필터링
# ===========================
print("="*80)
print("🔍 AKT/EMT 관련 유전자 필터링 시작")
print("="*80 + "\n")

# selected_genes_top.csv 로드
selected_genes_path = REPORTS_DIR / "selected_genes_top.csv"

if not selected_genes_path.exists():
    print(f"⚠️ 파일이 없습니다: {selected_genes_path}")
    print("먼저 train.py를 실행하여 selected_genes_top.csv를 생성하세요.")
else:
    genes_df = pd.read_csv(selected_genes_path)
    
    # AKT/EMT 관련 키워드 정의
    akt_keywords = [
        "AKT", "PIK3", "PTEN", "MTOR", "GSK3", "TSC1", "TSC2", 
        "PDPK1", "PDK1", "FOXO", "BAD", "MDM2", "IRS"
    ]
    
    emt_keywords = [
        "CDH1", "ECAD", "E-CADHERIN",  # Epithelial marker
        "VIM", "VIMENTIN",  # Mesenchymal marker
        "SNAI", "SNAIL", "SLUG", "TWIST", "ZEB",  # EMT transcription factors
        "FN1", "FIBRONECTIN",  # ECM
        "MMP", "MATRIX",  # Matrix metalloproteinase
        "TGFB", "TGF-B", "SMAD",  # TGF-beta pathway
        "WNT", "CTNNB", "CATENIN",  # Wnt pathway
        "NOTCH", "JAG", "DLL",  # Notch pathway
        "KRT", "KERATIN",  # Keratin (구조 관련, Enrichment에서 나온 것)
        "CDH2", "NCAD", "N-CADHERIN"  # Mesenchymal marker
    ]
    
    # 필터링 함수
    def filter_by_keywords(df, keywords, label):
        pattern = "|".join(keywords)
        filtered = df[df["gene"].str.contains(pattern, case=False, na=False)]
        filtered = filtered.copy()
        filtered["category"] = label
        return filtered
    
    # AKT 관련 유전자 필터링
    akt_genes = filter_by_keywords(genes_df, akt_keywords, "AKT_pathway")
    print(f"🔹 AKT pathway 관련 유전자: {len(akt_genes)}개")
    if len(akt_genes) > 0:
        print(akt_genes[["gene", "coef", "abs_coef", "category"]].head(10).to_string(index=False))
    else:
        print("   → AKT pathway 관련 유전자가 선택 유전자(300개)에 포함되지 않음")
    print()
    
    # EMT 관련 유전자 필터링
    emt_genes = filter_by_keywords(genes_df, emt_keywords, "EMT_related")
    print(f"🔹 EMT 관련 유전자: {len(emt_genes)}개")
    if len(emt_genes) > 0:
        print(emt_genes[["gene", "coef", "abs_coef", "category"]].head(20).to_string(index=False))
    else:
        print("   → EMT 직접 마커 유전자가 선택 유전자(300개)에 포함되지 않음")
    print()
    
    # 통합 결과
    combined = pd.concat([akt_genes, emt_genes], ignore_index=True)
    combined = combined.drop_duplicates(subset=["gene"]).sort_values("abs_coef", ascending=False)
    
    if len(combined) > 0:
        combined_path = REPORTS_DIR / "akt_emt_candidate_genes.csv"
        combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"✅ AKT/EMT 후보 유전자 저장: {combined_path}")
        print(f"   총 {len(combined)}개 유전자")
        print("\n[상위 10개]")
        print(combined[["gene", "coef", "abs_coef", "category"]].head(10).to_string(index=False))
    else:
        print("⚠️ AKT/EMT 관련 유전자가 선택 유전자(300개)에 포함되지 않음")
        print("\n💡 해석:")
        print("   - 선택된 300개 유전자는 '처리군 vs 대조군 분류'에 기여하는 유전자들")
        print("   - AKT/EMT 대표 마커가 직접 선택되지 않았더라도,")
        print("   - Enrichment에서 'Keratin filament', 'Cell periphery' 등")
        print("   - EMT와 간접적으로 연관된 경로가 enrichment됨")
        print("   - 이는 약물 처리가 EMT-like 구조 변화에 영향을 줄 가능성 시사")


print("\n" + "="*80)
print("✅ 모든 표 및 필터링 완료!")
print("="*80)