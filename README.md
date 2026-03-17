# ngs-moa-classifier

RNA-seq gene expression 기반:
- 처리군(12b) vs 대조군(Control) 이진 분류
- LogisticRegression(ElasticNet)로 feature selection 후
- RandomForest + LogisticRegression soft voting

## 1) 환경
- Python 3.10.6 권장

## 2) 설치
```bash
python -m venv .venv
# windows
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
