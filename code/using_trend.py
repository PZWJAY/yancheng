import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('../data/[new] yancheng_train_20171226.csv')

df_train_sub = df_train[(df_train['class_id'] == 103507)].groupby(['sale_date']).sale_quantity.sum().round()
a = 0