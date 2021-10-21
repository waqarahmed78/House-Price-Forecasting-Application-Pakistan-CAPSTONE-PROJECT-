import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('PakistanScrapper.csv',index_col=6)
df.drop(['Unnamed: 0'],axis=1,inplace=True)

df['Bathrooms'] = df['Bathrooms'].fillna(df['Bathrooms'].median())
df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].median())

df.drop(['URL','Img_url','Title','Phone','Purpose','ContactName','ShortDescription'],axis='columns',inplace=True)
df.drop(['Latitude','Longitude'],axis='columns',inplace=True)

df.drop(['Location'],axis=1,inplace=True)
X = df[['Bathrooms', 'Bedrooms','Rooms','Area']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 1)
from sklearn.ensemble import RandomForestRegressor
rfr_model = RandomForestRegressor(n_estimators=100,random_state=21,max_depth = 6,min_samples_split=2) #further hyper
rfr_model.fit(X_train,y_train)


# xg_model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
# xg_model.fit(X_train,y_train)

pickle.dump(rfr_model,open('rfrmodel.pkl','wb'))
