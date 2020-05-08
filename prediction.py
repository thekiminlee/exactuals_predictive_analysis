from xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# loading model
model = XGBClassifier()
model.load_model('exactuals_001.model')

# accepting user data
df = pd.read_csv('test.csv')

df = df.drop(columns=['payee_rating'], axis=1)
df = df.drop(columns=['Unnamed: 0'], axis=1)

le = LabelEncoder()
for attr in df:
    if(df[attr].dtype == 'object'):
        df.loc[:,(attr)] = le.fit_transform(df.loc[:,(attr)])


print(df)

prediction = model.predict(df)
print(prediction)
