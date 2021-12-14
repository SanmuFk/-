import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew

#读取数据
train_data = pd.read_csv("F:/机器学习/train.csv")
test_data = pd.read_csv("F:/机器学习/test.csv")

#显示数据
train_data.head()
test_data.head()

#查看数据类型
train_data.dtypes.value_counts()

# 保存数据第一行id
train_data_ID = train_data['Id']
test_data_ID = test_data['Id']
#删除id列
train_data.drop("Id", axis = 1, inplace = True)
test_data.drop("Id", axis = 1, inplace = True)

#查看居住面积与售价对应的散点图
fig, ax = plt.subplots() 
ax.scatter(x = train_data['GrLivArea'], y = train_data['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
# 由上图散点图分布，去掉面极端值
train_data = train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index)

#查看预测样本价格是否符合正态分布并修正
#SalePrice分布图
sns.distplot(train_data['SalePrice'] , fit=norm)
#SalePrice均值、均方差
(mu, sigma) = norm.fit(train_data['SalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()

#验证
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)
plt.show()

#转化
train_data["SalePrice"] = np.log1p(train_data["SalePrice"])

#使用分布图再次查看变量是否达到正态分布
sns.distplot(train_data['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train_data['SalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()

#再次验证
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)
plt.show()

#连接两个数据集
ntrain = train_data.shape[0]
ntest = test_data.shape[0]
y_train = train_data.SalePrice.values
#连接两个表，将两个表的数据共同处理
all_data = pd.concat((train_data, test_data)).reset_index(drop=True)
#去掉目标变量SalePrice列
all_data.drop(['SalePrice'], axis=1, inplace=True)
#生成数据形状大小
print("first all_data size is : {}".format(all_data.shape))

#通过热力图来判断相关性
corrmat = train_data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()

#构造函数
def MissingValue(df):
    miss_value =df.isnull().sum()
    miss_percentage = miss_value / df.shape[0]
    miss_df = pd.concat([miss_value,miss_percentage],axis=1)
    miss_df = miss_df.rename(columns={0:'MissingValue',1:'%MissingPercent'})
    miss_df = miss_df.loc[miss_df['MissingValue']!=0,:]
    miss_df = miss_df.sort_values(by='%MissingPercent',ascending = False)
    return miss_df

#查看缺失值
MissingValue(train_data)
MissingValue(test_data)

#填补缺失值
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'):
    all_data[col].fillna("None", inplace=True)
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
'BsmtHalfBath', 'MasVnrArea'):
    all_data[col].fillna(0, inplace=True)
for col in ('MSZoning', "Functional", 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'MSSubClass'):
    all_data[col].fillna(all_data[col].mode()[0], inplace=True)
all_data.drop(['Utilities'], axis=1, inplace=True)
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


#检测是否还存在缺失值
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print("missing_data:",missing_data.head())

#数值变为类别
#将这几个列的属性变为str类型，才可以进行下面编码处理
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

#进行标签编码
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))
    
#组合特征
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

#将数值特征提取
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

#显示偏斜度，进行降序排序
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print("skew:",skewness.head(10))

#log转换
skewness = skewness[abs(skewness.Skew) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
print("LabelEncoder：all_data.shape:", all_data.shape)

#将属性特征转化为指示变量
all_data = pd.get_dummies(all_data)
print("one-hot：all_data.shape:", all_data.shape)

#数据预处理完成后的总训练集、预测集
train_data = all_data[:ntrain]
test_data = all_data[ntrain:]

