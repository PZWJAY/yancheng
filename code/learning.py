import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm #norm distribution
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('../data/preprocessed/precrocessed2.csv')
df_train_sub = df_train.groupby(['sale_date','class_id']).sale_quantity.sum().round()
#sns.distplot(df_train_sub)
#plt.show()
#print('Skewness: %f' % df_train_sub.skew()) #偏度，衡量分布不对称程度
#print('Kurtosis: %f' % df_train_sub.kurt()) #峰度，衡量分布曲线尖峭程度

'''
#-----------------------------------------------------------------------------------------------------------------------------------------
#print dataset imformation
print(df_train.describle())
#-----------------------------------------------------------------------------------------------------------------------------------------
'''

'''
#-----------------------------------------------------------------------------------------------------------------------------------------
#numerical type analyse with 'sale_quantity' using scatter
var = 'class_id'
data = pd.concat([df_train['sale_quantity'],df_train[var]],axis=1)
data.plot.scatter(x=var,y='sale_quantity',ylim=(0,8000)
plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------
'''

'''
#-----------------------------------------------------------------------------------------------------------------------------------------
#categorical type analyse with 'sale_quantity' using box figure
var = 'sale_date'
data = pd.concat([df_train['sale_quantity'],df_train[var]],axis=1)
fig = sns.boxplot(x=var,y='sale_quantity',data=data)
fig.axis(ymin=0,ymax=8000)
plt.xticks(rotation=90)
plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------
'''
#-----------------------------------------------------------------------------------------------------------------------------------------
#overall correlation
corrmat = df_train.corr()
sns.heatmap(corrmat,vmax=8,square=True)
plt.xticks(rotation=90)
plt.yticks(rotation=360)
plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------
#'sale_quantity' correlation
k = 32 #number of variables for heatmap
cols = corrmat.nlargest(k,'sale_quantity')['sale_quantity'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)
plt.xticks(rotation=90)
plt.yticks(rotation=360)
plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------

sns.set()
cols = ['sale_quantity','sale_date','compartment','type_id','price_level']
sns.pairplot(df_train[cols],size=2.5)
plt.show()
a = 0
