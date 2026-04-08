import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('placement.csv')
df = df.iloc[:, 1:]

x = df.iloc[:, 0:2]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.values)
x_test_scaled  = scaler.transform(x_test.values)

clf = LogisticRegression()
clf.fit(x_train_scaled, y_train)

pickle.dump(clf,    open('model.pkl',  'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("✅ model.pkl and scaler.pkl saved!")
