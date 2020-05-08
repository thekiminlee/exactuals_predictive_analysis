import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


class Exactuals():
    def __init__(self, _file, _label):
        self.df = pd.read_csv(_file)
        self.df = self.df.drop(columns=["Unnamed: 0"], axis=1)

        le = LabelEncoder()
        for attr in self.df:
            if(self.df[attr].dtype == 'object'):
                self.df.loc[:,(attr)] = le.fit_transform(self.df.loc[:,(attr)])

        self.feature = self.df.drop(columns=[_label], axis=1)
        self.label = self.df[_label]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.feature, self.label, test_size=0.3)
        
        self.prediction = None
        self.model = XGBClassifier()

    def train(self):
        print("Training XGB model...")
        self.model.fit(self.X_train, self.y_train)
        print("Training complete")

    def evaluate(self):
        cvs = cross_val_score(estimator=self.model, X=self.X_train, y=self.y_train, cv=10, n_jobs=-1)
        print("Accuracy: ", round(cvs.mean()*100, 2), "%")

    def export(self, name):
        self.model.save_model(name)

if __name__ == '__main__':
    exactuals = Exactuals('sample.csv', 'payee_rating')
    exactuals.train()
    exactuals.evaluate()
    exactuals.export('exactuals_001.model')
