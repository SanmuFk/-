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
#选择线性回归模型
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
model_LinearRegression = LinearRegression()
model_RidgeCV = RidgeCV() # 岭回归L1
model_LassoCV = LassoCV() # Lasso回归L2
model_ElasticNetCV = ElasticNetCV() # L1、L2组合

model = model_ElasticNetCV
try_different_method(model)
