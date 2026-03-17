import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

# 1. 경로 설정 최적화
current_file = Path(__file__).resolve()
# 프로젝트 루트(ngs-moa-classifier) 폴더를 찾을 때까지 위로 올라감
project_root = current_file
for parent in current_file.parents:
    if parent.name == "ngs-moa-classifier":
        project_root = parent
        break

# 예측 결과 파일이 저장된 정확한 경로 (핵심!)
results_path = project_root / "data" / "processed" / "external_test_results.csv"
figures_path = project_root / "reports" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

def visualize_external_results():
    print(f"🔍 확인 중인 경로: {results_path}")
    
    # 1. 결과 데이터 로드
    if not results_path.exists():
        print("❌ 에러: 결과 파일을 찾을 수 없습니다!")
        print(f"찾으려던 위치: {results_path}")
        
        # 실제 어디에 파일이 있는지 주변 탐색 (디버깅용)
        search_base = project_root / "data"
        print(f"💡 현재 data 폴더 내 구조:")
        for root, dirs, files in os.walk(search_base):
            for file in files:
                if "external_test_results" in file:
                    print(f"Found it here: {os.path.join(root, file)}")
        return
    
    df = pd.read_csv(results_path)
    print("✅ 데이터 로드 성공! 시각화를 시작합니다.")

    # 그래프 스타일 설정
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # --- 그래프 1: 확률 분포 ---
    # 컬럼명이 MoA_Probability인지 확인 후 시각화
    col_name = 'MoA_Probability' if 'MoA_Probability' in df.columns else 'MoA_Prob'
    sns.histplot(df[col_name], bins=15, kde=True, ax=ax[0], color='skyblue')
    ax[0].set_title('Distribution of MoA Probabilities', fontsize=12)
    ax[0].set_xlabel('Probability (0: Control, 1: MoA)')

    # --- 그래프 2: 클래스별 샘플 수 ---
    sns.countplot(x='Prediction', data=df, ax=ax[1], palette='Set2', hue='Prediction', legend=False)
    ax[1].set_title('Predicted Class Counts', fontsize=12)

    plt.suptitle(f"GSE100928 External Validation Results", fontsize=16)
    
    # 저장
    save_file = figures_path / "external_test_visualization.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_file)
    print(f"📊 시각화 완료! 저장 위치: {save_file}")
    plt.show()

if __name__ == "__main__":
    visualize_external_results()