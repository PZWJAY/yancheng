from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/preprocessed/precrocessed.csv")
res = pd.read_csv("../data/yancheng_testA_20171225.csv")
res_df = res.copy()
res_df = pd.merge(res_df[['predict_date','class_id']],df,how='left',on=['class_id'])
res_df = res_df.drop('sale_date',axis=1)
res_df.rename(columns={'predict_date':'sale_date'},inplace = True)


#print(df.sale_quantity.skew())

df_y = df['sale_quantity']
df_x = df.drop('sale_quantity',axis=1)

#train_x,test_x,train_y,test_y = train_test_split(df_x,df_y,test_size=0.3)

xgb_params = {
    'eta' : 0.05,
    'booster': 'gbtree',
    'max_depth' : 8,
    'subsample' : 0.7,
    'colsample_bytree' : 0.7,
    'objective' : 'reg:linear',
    'eval_metric' : "rmse",
    'silent' : 0,
}

train_y_log = np.log(df_y)
#test_y_log = np.log(test_y)

dtrain_log = xgb.DMatrix(df_x,train_y_log)
#dvalid_log = xgb.DMatrix(test_x,test_y_log)
#watchlist = [(dtrain_log,'train'),(dvalid_log,'eval')]

#print(train_y_log.skew())

#cv_output_log = xgb.cv(xgb_params,dtrain_log,num_boost_round=10000,early_stopping_rounds=5000,verbose_eval=50,show_stdv=False)
#num_boost_rounds = len(cv_output_log)


model = xgb.train(xgb_params,dtrain_log,num_boost_round=100000)

predict = res_df.drop('sale_quantity',axis=1)
dpredict = xgb.DMatrix(predict)
pred_quantity_log = model.predict(dpredict)
pred_quantity = np.exp(pred_quantity_log)
sub_ = pd.DataFrame({u'sale_quantity':pred_quantity})
res_df['sale_quantity'] = sub_['sale_quantity']

result = res_df.groupby(['class_id']).sale_quantity.sum().round()
predict = result.reset_index()
result = pd.merge(res[['predict_date','class_id']],predict,how='left',on=['class_id'])
result = result.fillna(0)
result.columns = ['predict_date','class_id','predict_quantity']
result.to_csv("../result/xgb0125_without_watchlist.csv",index=False,header=True)


#fig,ax = plt.subplot(1,1,figsize=(8,13))
#xgb.plot_importance(model,max_num_features=50,height=0.5,ax=ax)
#plt.show()