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
    
#具体模型选择
#我首先使用决策树回归模型
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()

model = model_DecisionTreeRegressor
try_different_method(model)
