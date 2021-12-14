import numpy as np
import pandas as pd
import preprocessing
import train-DecisionTreeRegressor #使用不同模型在这修改接口

# 训练模型
model = train-DecisionTreeRegressor.model
# 预测数据集
test = preprocessing.test_data
# 预测数据id列
test_ID = preprocessing.test_data_ID

# 模型预测
predictions = model.predict(test)

# 对预测结果expm1反转，因为之前做过正态处理
final_predictions = np.expm1(predictions)

# 将结果存入submission.csv中
submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = final_predictions
submission.to_csv('submission.csv', index=False)
