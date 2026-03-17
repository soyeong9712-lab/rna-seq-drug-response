import pandas as pd
from src.utils.paths import RAW_DIR

def load_gse100928():
    """
    GEO GSE100928 데이터를 읽어 데이터프레임으로 반환
    """
    file_path = RAW_DIR / "GSE100928_series_matrix.txt"
    
    # '!series_matrix_table_begin' 문자열이 나오는 줄부터 데이터 시작
    skip_rows = 0
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith("!series_matrix_table_begin"):
                skip_rows = i + 1
                break
                
    # 데이터 로딩 (탭 구분자)
    df = pd.read_csv(file_path, sep='\t', skiprows=skip_rows, comment='!')
    df = df.dropna(subset=['ID_REF'])
    df = df[df['ID_REF'] != "!series_matrix_table_end"]
    
    # 인덱스를 유전자 ID(ID_REF)로 설정
    df = df.set_index('ID_REF')
    return df