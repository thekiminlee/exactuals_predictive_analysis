from xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

models = ['exactuals_001_label1.model', 'exactuals_001_label2.model', 'exactuals_001_label3.model']
labels = ['payee_satisfaction', 'payor_satisfaction', 'overall_satisfaction']

# loading model
df = pd.read_csv('test.csv')
df = df.drop(columns=["Unnamed: 0", "transaction_start_date", "transaction_end_date"]+labels, axis=1)

for i in range(0, 3):
    model = XGBClassifier()
    model.load_model(models[i])
    prediction = model.predict(df)
    print(labels[i], prediction)

