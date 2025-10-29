# our rec

## 安装依赖
`pip install -r requirements.txt`

若是中国用户，建议：
1. `pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1`
2. 注释掉 `torch==2.5.1+cu121`
3. `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`