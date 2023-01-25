import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Quiz 2
# data = pd.read_csv("https://raw.githubusercontent.com/doguilmak/Heart-Diseaseor-Attack-Classification/main/heart_disease_health_indicators_BRFSS2015.csv")
# print(data.head())
# y = data['HeartDiseaseorAttack']
# x = data.drop('HeartDiseaseorAttack', axis=1)
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
# logistic_model = LogisticRegression(max_iter=1000000)
# logistic_model.fit(x_train, y_train)
# print(logistic_model.score(x_test, y_test))
#
# gaussian_model = GaussianNB(priors=[0.4, 0.6])
# gaussian_model.fit(x_train, y_train)
# print(gaussian_model.score(x_test, y_test))
#
# x_new = data[['Fruits', 'PhysActivity', 'HvyAlcoholConsump']]
# bernouli_model = BernoulliNB()
# x1_train, x1_test, y1_train, y1_test = train_test_split(x_new, y, random_state=1)
# bernouli_model.fit(x1_train, y1_train)
# print(bernouli_model.score(x1_test, y1_test))


# Quiz 3

# data = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
# # print(data.head())
# label = LabelEncoder()
# data['sex'] = label.fit_transform(data['sex'])
# data['smoker'] = label.fit_transform(data['smoker'])
# data['region'] = label.fit_transform(data['region'])
#
# y = data['charges'].values
# x = data.drop('charges', axis=1).values
#
# model = LinearRegression()
# model.fit(x, y)
# print(model.score(x, y))
#
# hybrid_pip = Pipeline(steps=[("scaler", RobustScaler()), ("pca", PCA(n_components=1)), ("algo", Lasso())])
# hybrid_pip.fit(x, y)
# # print(hybrid_pip.score(x, y))
# print(hybrid_pip.named_steps['pca'].explained_variance_ratio_)


# Quiz 4
# data = pd.read_csv("https://raw.githubusercontent.com/kvinlazy/Dataset/master/drug200.csv")
# print(data.head())

# label = LabelEncoder()
# data['Sex'] = label.fit_transform(data['Sex'])
# data['BP'] = label.fit_transform(data['BP'])
# data['Cholesterol'] = label.fit_transform(data['Cholesterol'])
#
# y = data['Drug'].values
# x = data.drop('Drug', axis=1).values
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.1)
# my_model = AdaBoostClassifier()
# my_model.fit(x_train, y_train)
# print(my_model.score(x_test, y_test))
#
# params = {"learning_rate": [0.1, 0.2, 0.3, 0.4, 0.7, 0.9]}
# new_params = {"learning_rate": [0.1, 0.2, 0.3, 0.4, 0.7, 0.9], 'n_estimators': [30, 40, 60]}
# hybrid = GridSearchCV(my_model, params, scoring='accuracy', cv=3, n_jobs=1)
# new_hybrid = GridSearchCV(my_model, new_params, scoring='accuracy', cv=3, n_jobs=1)
# hybrid.fit(x, y)
# new_hybrid.fit(x, y)
# print(hybrid.best_params_, hybrid.best_score_)
# print(new_hybrid.best_params_, new_hybrid.best_score_)


# Quiz 5

# data = pd.read_csv("credit.csv")
# print(data.head())
# y = data['y']
# x = data.drop('y', axis=1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
# svc_model = SVC()
# svc_model.fit(x_train, y_train)
# print(svc_model.score(x_test, y_test))
# 
# selector = SelectKBest(score_func=f_classif, k=6)
# selector.fit(x, y)
# print(selector.get_feature_names_out())


data = pd.read_csv("regression.csv")
print(data.head())
y = data['x'].values
x = data.drop('x', axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.15)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
my_model = Lasso()
my_model.fit(x_train, y_train)
print(my_model.score(x_test, y_test))
