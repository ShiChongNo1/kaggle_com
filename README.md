# Kaggle 比赛

## 一. elo-merchant: 
### 数据集（https://www.kaggle.com/c/elo-merchant-category-recommendation）
#### 1. 模型：二阶段建模（一阶段分类二阶段预测）
    - 目录 ：meching_Lear ing/mian.py (数据预处理也在此阶段)
    - 结果排名 Top 1% (40名左右)
    - 一、二阶段建模（LGBM + CatBoost + XGBoost)
#### 2. 模型：简单的线性神经网络
    - 目录：nerul/linear-module.py (模型未收敛，判断原因为：模型设计过于简单)
#### 3. 模型：transformer
    -目录：transformer/run 
    -结果排名：