import os
import pickle
import urllib
from io import BytesIO

import pandas as pd
from numpy import int64
from shap import benchmark
import numpy as np
import shap
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

device = "cpu"
from torch import nn

url_base = "http://localhost:8000/"


def get_web_model(url, is_pickle=False):
    url = url_base + url

    with urllib.request.urlopen(url) as f:
        buf = BytesIO(f.read())
        f.close()
        if not is_pickle:
            model = torch.jit.load(buf)
        else:
            model = pickle.load(buf)
        return model


class MyNetWork(nn.Module):
    def __init__(self):
        super(MyNetWork, self).__init__()
        self.fc1 = nn.Linear(5, 1)
        # self.fc2=nn.Linear(2,1)

    def forward(self, x):
        x = self.fc1(x)
        # x=self.fc2(x)
        return x

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

class Income():
    def __init__(self, filenames, net_path):
        self.models = []
        print(filenames)
        for file_path in filenames:
            self.models.append(get_web_model(file_path, is_pickle=True))
        self.net = MyNetWork()
        with urllib.request.urlopen(url_base + net_path) as f:
            buf = BytesIO(f.read())
            f.close()
            self.net.load_state_dict(torch.load(buf, map_location=device))
        self.masker = shap.maskers.Independent(self.get_X_train())

    def get_X_train(self):
        df = pd.read_csv("adult.csv", encoding='latin-1')
        df[df == '?'] = np.nan
        for col in ['workclass', 'occupation', 'native.country']:
            df[col].fillna(df[col].mode()[0], inplace=True)
        X = df.drop(['income'], axis=1)
        self.X=X
        y = df['income']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex',
                       'native.country']
        self.les=[]
        for feature in self.categorical:
            le = preprocessing.LabelEncoder()
            X_train[feature] = le.fit_transform(X_train[feature])
            self.les.append(le)
        self.scaler = StandardScaler()
        X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X.columns)

        return X_train
    def prepocess(self,data):
        for index,feature in enumerate(self.categorical):
            data[feature] = self.les[index].transform(data[feature])
        data = pd.DataFrame(self.scaler.transform(data), columns=self.X.columns)
        return data
    def predict(self, X):
        X=self.prepocess(X.copy())
        ys = []
        for loaded_model in self.models:
            ys.append(loaded_model.predict(X)[0])
        return np.argmax(np.bincount(ys))

    def logit_predict(self, X):
        ys = []
        for loaded_model in self.models:
            ys.append(loaded_model.predict(X))
        ans = np.array(ys)
        return np.array(pd.DataFrame(ans).mode()).squeeze()
    def all_logit_predict(self,X):
        ys=[]
        for i in findAllFile("E:\Income\models"):
            name = os.path.basename(i)
            if name.startswith("clf"):
                loaded_model = pickle.load(open(i, 'rb'))
                ys.append(loaded_model.predict(X))
        ans = np.array(ys)
        res=np.array(pd.DataFrame(ans).mode())[0]
        return res


    def explain(self, X,method):
        X = self.prepocess(X.copy())
        ys = []
        for loaded_model in self.models:
            explainer = shap.explainers.Tree(loaded_model, self.masker)
            ys.append(explainer(X).values)
        if method==0:
            output=np.array(
            self.net(torch.tensor(np.array(ys).transpose([1, 2, 0])).to(torch.float)).squeeze(dim=0).transpose(1,
                                                                                                               0).detach().numpy())

            return output
        elif method>=1 and method<=5:
            return np.array(ys[method-1])
        else:
            return np.mean(ys,axis=0)

    def eval_exp(self,X,method,all=False):
        if not all:
            data = self.prepocess(X.copy())
            smasker = shap.benchmark.SequentialMasker(
                "keep", "positive", self.masker, self.logit_predict, np.array(data)
            )
            exp = self.explain(X, method)
            keep_positive_value = smasker(exp, name="Tree").value
            smasker = shap.benchmark.SequentialMasker(
                "keep", "negative", self.masker, self.logit_predict, np.array(data)
            )
            keep_negative_value = smasker(exp, name="Tree").value
            return {"keep_positive": keep_positive_value, "keep_negative": keep_negative_value}
        else:
            data = self.prepocess(X.copy())
            smasker = shap.benchmark.SequentialMasker(
                "keep", "positive", self.masker, self.all_logit_predict, np.array(data)
            )
            exp = self.explain(X, method)
            keep_positive_value = smasker(exp, name="Tree").value
            smasker = shap.benchmark.SequentialMasker(
                "keep", "negative", self.masker, self.all_logit_predict, np.array(data)
            )
            keep_negative_value = smasker(exp, name="Tree").value
            return {"keep_positive": keep_positive_value, "keep_negative": keep_negative_value}

