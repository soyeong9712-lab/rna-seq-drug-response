import torch
import torch.nn as nn

class GeneExpressionDNN(nn.Module):
    """
    RNA-seq 데이터의 고차원 피처를 처리하기 위한 심층 신경망
    """
    def __init__(self, input_dim):
        super(GeneExpressionDNN, self).__init__()
        # 피처(유전자) 수가 많으므로 레이어를 깊게 쌓고 Dropout으로 과적합 방지
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),     # 30% 노드 생략 (과적합 방지)
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1),    # 이진 분류 (0: 대조군, 1: 처리군)
            nn.Sigmoid()         # 확률값 출력
        )

    def forward(self, x):
        return self.model(x)