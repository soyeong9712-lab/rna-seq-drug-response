import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
import numpy as np
from src.models.dnn_model import GeneExpressionDNN
from src.features.preprocess import load_processed, log2_transform
from src.utils.paths import PROCESSED_DIR, MODELS_DIR

def main():
    # 1. 기존 전처리 데이터 로드
    X, y = load_processed(
        PROCESSED_DIR / "X_gene_expression.csv",
        PROCESSED_DIR / "y_labels.csv"
    )
    X_log = log2_transform(X)
    
    # 디버깅: 데이터 확인
    print(f"X_log shape: {X_log.shape}")
    print(f"y shape: {y.shape}")
    print(f"X_log 통계: min={X_log.min().min():.2f}, max={X_log.max().max():.2f}, mean={X_log.mean().mean():.2f}")
    print(f"y 값: {y.values}")
    print(f"NaN 확인: X={X_log.isna().sum().sum()}, y={y.isna().sum()}")
    
    # NaN/Inf 처리
    X_log = X_log.fillna(0)
    X_log = X_log.replace([np.inf, -np.inf], 0)
    
    # 정규화 (StandardScaler 대신 간단한 정규화)
    X_mean = X_log.mean(axis=0)
    X_std = X_log.std(axis=0) + 1e-8  # 0으로 나누기 방지
    X_normalized = (X_log - X_mean) / X_std

    # 2. 모델 설정
    input_dim = X_normalized.shape[1]
    model = GeneExpressionDNN(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 학습률 낮춤
    criterion = torch.nn.BCELoss()
    
    # 3. 학습
    X_tensor = torch.FloatTensor(X_normalized.values)
    y_tensor = torch.FloatTensor(y.values).view(-1, 1)
    
    # y 값이 0~1 범위인지 확인
    print(f"y_tensor 범위: min={y_tensor.min():.2f}, max={y_tensor.max():.2f}")
    
    print("\n🚀 딥러닝 모델 학습 시작...")
    for epoch in range(200):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (gradient explosion 방지)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")
            
    # 4. 모델 저장
    torch.save(model.state_dict(), MODELS_DIR / "dnn_model.pth")
    print("✅ 모델 저장 완료: models/dnn_model.pth")

if __name__ == "__main__":
    main()