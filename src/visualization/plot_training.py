import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
from src.models.dnn_model import GeneExpressionDNN
from src.features.preprocess import load_processed, log2_transform
from src.utils.paths import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR

def plot_results():
    # 1. 데이터 및 모델 로드
    X, y = load_processed(PROCESSED_DIR / "X_gene_expression.csv", PROCESSED_DIR / "y_labels.csv")
    X_log = log2_transform(X)
    
    input_dim = X_log.shape[1]
    model = GeneExpressionDNN(input_dim)
    model.load_state_dict(torch.load(MODELS_DIR / "dnn_model.pth"))
    model.eval()

    # 2. 예측값 계산
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_log.values)
        predictions = model(X_tensor).numpy().flatten()
    
    # 3. 시각화
    plt.figure(figsize=(10, 5))
    
    # 실제값 vs 예측값 산점도
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(y)), y.values, color='blue', label='Actual', s=100)
    plt.scatter(range(len(predictions)), predictions, color='red', marker='x', label='Predicted', s=100)
    plt.title("Actual vs Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Probability (0: Control, 1: MoA)")
    plt.legend()

    # 결과 저장
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_result.png")
    print(f"📊 시각화 완료: {FIGURES_DIR}/training_result.png")
    plt.show()

if __name__ == "__main__":
    plot_results()