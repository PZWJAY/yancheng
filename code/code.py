import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#import seaborn as sns

#import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import model_selection,preprocessing

#sns.set(style="white",color_codes=True)


df = pd.read_csv('../data/[new] yancheng_train_20171226.csv')
test = pd.read_csv('../data/yancheng_testA_20171225.csv')
train = df.copy()

group_id = train[(train.sale_date==201710)].groupby(['class_id']).sale_quantity.sum().round()
group_id0 = train[(train.sale_date==201709)].groupby(['class_id']).sale_quantity.sum().round()
predict = group_id.reset_index()
predict0 = group_id0.reset_index()
#predict['sale_quantity'] = predict['sale_quantity'].map(lambda x:int(x+30))
result = pd.merge(test[['predict_date','class_id']],predict,how='left',on=['class_id'])
result = result.fillna(0)
result0 = pd.merge(test[['predict_date','class_id']],predict0,how='left',on=['class_id'])
result0 = result0.fillna(0)
result.columns = ['predict_date','class_id','predict_quantity']
result0.columns = ['predict_date','class_id','predict_quantity']
result0['predict_quantity_1'] = result['predict_quantity'] - result0['predict_quantity']
result0['predict_quantity_1'] = result0['predict_quantity_1'].map(lambda x:x if x > 0 else 0)

result.to_csv('../result/predict201710_plus_with_201709_difference.csv',index=False,header=True)

'''
tmp = 201201
tmp0 = 201201
for i in range(1,73):
    group_date = train[(train.sale_date == tmp)].groupby(['class_id']).sale_quantity.sum().round()
    group_date_series = group_date.reset_index()
    sns.barplot(x='class_id',y='sale_quantity',data=group_date_series)
    loc = '../plot/'+str(tmp)+ '.png'
    plt.savefig(loc)
    tmp += 1
    if i % 12 == 0:
        tmp = tmp0 + 100
        tmp0 += 100
'''
#group_id = train.groupby(['class_id']).sale_quantity.sum().round()
#group_id_series = group_id.reset_index()
#sns.boxplot(x='sale_date',y='sale_quantity',data=train)
#plt.show()

#分离sale_date，分为销售年份和月份
#train['sale_date_year'] = train['sale_date'].map(lambda x:int(x/100))
#train['sale_date_month'] = train['sale_date'].map(lambda x:x%100)

#lbl = preprocessing.LabelEncoder()
#lbl.fit(list(train['gearbox_type'].values))
#train['gearbox_type'] = lbl.transform(list(train['gearbox_type'].values))
'''
tmp0 = train[(train.sale_date_year == 2017)].groupby(['sale_date_month','gearbox_type']).sale_quantity.sum().round()
tmp0 = tmp0.reset_index()
ax = sns.barplot(x='sale_date_month',hue='gearbox_type',y='sale_quantity',data=tmp0)
ax.set_title('2017 different gearbox_type sale quantity')
plt.show()
'''
#deal with missing value
for index,row in train.iterrows():
    ele = ['compartment', 'type_id', 'level_id', 'department_id', 'TR', 'gearbox_type', 'displacement', 'if_charging',
           'price_level',
           'driven_type_id', 'newenergy_type_id', 'emission_standards_id', 'if_MPV_id', 'if_luxurious_id', 'power',
           'cylinder_number',
           'engine_torque', 'car_length', 'car_width', 'car_height', 'total_quality', 'equipment_quality',
           'rated_passenger',
           'wheelbase', 'front_track', 'rear_track']
    if row.fuel_type_id == '-':
        val = 0
        max_proximity = 0
        tmpdf = train[train['class_id'] == row.class_id]
        for index2,row2 in tmpdf.iterrows():
            if index == index2 or row2.fuel_type_id == '-':
                continue
            cnt = 0
            for e in ele:
                if row[e] == row2[e]:
                    cnt += 1
            if cnt > max_proximity:
                max_proximity = cnt
                val = int(row2.fuel_type_id)
        #row.fuel_type_id = val
        train.loc[index:index,['fuel_type_id']] = val

    if row.price == '-':
        price = 0
        max_proximity = 0
        tmpdf = train[train['class_id'] == row.class_id]
        if len(set(tmpdf['price'].values)) == 1: #假如该类型的所有车的售价都缺失
            if row.price_level[-1] == 'L':
                price = 5
            else:
                tmp_str = row.price_level
                tmp_str = tmp_str.split('-')
                price = (int(tmp_str[0]) + int(tmp_str[1][:-1])) / 2
        else:
            for index2,row2 in tmpdf.iterrows():
                cnt = 0
                if index2 == index or row2.price == '-':
                    continue
                for e in ele:
                    if row[e] == row2[e]:
                        cnt += 1
                if cnt > max_proximity:
                    max_proximity = cnt
                    price = row2.price
        train.loc[index:index,['price']] = price

price_level_dict = {'5WL':1,'5-8W':2,'8-10W':3,'10-15W':4,'15-20W':5,'20-25W':6,'25-35W':7,'35-50W':8,'50-75W':9}

train['fuel_type_id'] = train['fuel_type_id'].map(lambda x:int(x))
#train['rated_passenger_1'] = train['rated_passenger'].map(lambda x:int(str(x)[-1:]))
train['price'] = train['price'].map(lambda x:float(x))
train['level_id'] = train['level_id'].map(lambda x:int(x) if x != '-' else 0)
train['engine_torque'] = train['engine_torque'].map(lambda x: '0' if x == '-' else x)
train['price_level'] = train['price_level'].map(lambda x:int(price_level_dict[x]))

file = open('../analyse/columnsType2.txt','w')

#one hot encoding
for col in train.columns:
    tmp = col + '-' + str(train[col].dtype)
    file.write(tmp)
    tmp = set(train[col].values)
    tmp0 = '{'
    for el in tmp:
        tmp0 += str(el)
        tmp0 += ' '
    tmp0 += '}'
    file.write(tmp0)
    file.write('\n')
    if train[col].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[col].values))
        train[col] = lbl.transform(list(train[col].values))
file.close()
train.to_csv("../data/preprocessed/precrocessed.csv",index=False,header=True)

#print(train.head(3))