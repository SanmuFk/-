import numpy as np
import Preprocessing

#随机生成训练集和测试集
from sklearn.model_selection import train_test_split
train = train_data
y_train = y_train
X_train, X_test, y_train, y_test = train_test_split(
    train, y_train, random_state=42, test_size=.25)

#进入模型
from sklearn.metrics import mean_squared_error  #RMSE均方根误差
def try_different_method(model):
    model.fit(X_train,y_train)  #拟合模型
    score = model.score(X_test, y_test)  #模型评分
    result = model.predict(X_test)  #测试集结果（目标变量）
    
#使用其他模型进行优化
#选择LightGBM模型
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.01, n_estimators=3000,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

model = model_lgb
try_different_method(model)
