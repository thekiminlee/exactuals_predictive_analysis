import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


class Exactuals():
    def __init__(self, _file, _label1, _label2, _label3):
        self.df = pd.read_csv(_file)
        self.df = self.df.drop(columns=["Unnamed: 0", "transaction_start_date", "transaction_end_date"], axis=1)
    
        le = LabelEncoder()
        for attr in self.df:
            if(self.df[attr].dtype == 'object'):
                self.df.loc[:,(attr)] = le.fit_transform(self.df.loc[:,(attr)])

        self.feature = self.df.drop(columns=[_label1, _label2, _label3], axis=1)
        self.label1 = self.df[_label1]
        self.label2 = self.df[_label2]
        self.label3 = self.df[_label3]
        print(self.feature.head())
        # label 1
        self.label1_set = train_test_split(self.feature, self.label1, test_size=0.3)
        # label 2
        self.label2_set = train_test_split(self.feature, self.label2, test_size=0.3)
        # label 3
        self.label3_set = train_test_split(self.feature, self.label3, test_size=0.3)

        self.prediction = None
        self.model = XGBClassifier()

    def train(self, label_set):
        print("Training XGB model...")
        self.model.fit(label_set[0], label_set[2])
        pred = self.model.predict(label_set[1])
        print(pred, label_set[1])
        print("Training complete")

    def evaluate(self, label_set):
        cvs = cross_val_score(estimator=self.model, X=label_set[0], y=label_set[2], cv=10, n_jobs=-1)
        print("Accuracy: ", round(cvs.mean()*100, 2), "%")

    def export(self, name):
        self.model.save_model(name)

    def start(self,start):
        count = 1
        labels = [self.label1_set, self.label2_set, self.label3_set]
        for label in labels:
            self.train(label)
            self.evaluate(label)
            self.export('exactuals_'+ start + "_label" + str(count) + ".model")
            count += 1

if __name__ == '__main__':
    exactuals = Exactuals('sample.csv', 'payee_satisfaction','payor_satisfaction', 'overall_satisfaction')
    exactuals.start("001")
