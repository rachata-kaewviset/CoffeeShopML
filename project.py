import pandas as pd
import numpy as np
from LinearRegression import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('grouped_data.csv')
df = df.drop(['transaction_date'],axis=1)
df.replace('Monday', 0, inplace=True)
df.replace('Tuesday', 1, inplace=True)
df.replace('Wednesday', 2, inplace=True)
df.replace('Thursday', 3, inplace=True)
df.replace('Friday', 4, inplace=True)
df.replace('Saturday', 5, inplace=True)
df.replace('Sunday', 6, inplace=True)

df.replace('Astoria', 0, inplace=True)
df.replace("Hell's Kitchen", 1, inplace=True)
df.replace('Lower Manhattan', 2, inplace=True)

df.replace('Bakery', 0, inplace=True)
df.replace('Branded', 1, inplace=True)
df.replace('Coffee', 2, inplace=True)
df.replace('Coffee beans', 3, inplace=True)
df.replace('Drinking Chocolate', 4, inplace=True)
df.replace('Flavours', 5, inplace=True)
df.replace('Loose Tea', 6, inplace=True)
df.replace('Packaged Chocolate', 7, inplace=True)
df.replace('Tea', 8, inplace=True)

df.replace('Barista Espresso', 0, inplace=True)
df.replace('Biscotti', 1, inplace=True)
df.replace('Black tea', 2, inplace=True)
df.replace('Brewed Black tea', 3, inplace=True)
df.replace('Brewed Green tea', 4, inplace=True)
df.replace('Brewed herbal tea', 5, inplace=True)
df.replace('Clothing', 6, inplace=True)
df.replace('Drinking Chocolate', 7, inplace=True)
df.replace('Drip coffee', 8, inplace=True)
df.replace('Espresso Beans', 9, inplace=True)
df.replace('Gourmet Beans', 10, inplace=True)
df.replace('Gourmet brewed coffee', 11, inplace=True)
df.replace('Green beans', 12, inplace=True)
df.replace('Green tea', 13, inplace=True)
df.replace('Herbal tea', 14, inplace=True)
df.replace('Hot chocolate', 15, inplace=True)
df.replace('House blend Beans', 16, inplace=True)
df.replace('Housewares', 17, inplace=True)
df.replace('Organic Beans', 18, inplace=True)
df.replace('Organic Chocolate', 19, inplace=True)
df.replace('Organic brewed coffee', 20, inplace=True)
df.replace('Pastry', 21, inplace=True)
df.replace('Premium Beans', 22, inplace=True)
df.replace('Premium brewed coffee', 23, inplace=True)
df.replace('Regular syrup', 24, inplace=True)
df.replace('Scone', 25, inplace=True)
df.replace('Sugar free syrup', 26, inplace=True)
df.replace('Brewed Chai tea', 27, inplace=True)
df.replace('Chai tea', 28, inplace=True)

df.replace('Large', 0, inplace=True)
df.replace('Not Defined', 1, inplace=True)
df.replace('Regular', 2, inplace=True)
df.replace('Small', 3, inplace=True)

X = df.drop(['transaction_qty'], axis=1)
y = df['transaction_qty']

kf = KFold(n_splits=10, shuffle=True)
model = LinearRegression()

mse_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_absolute_error(y_test, y_pred)
    mse_scores.append(mse)

avg_mse = sum(mse_scores) / len(mse_scores)

print(f'Average absolute Squared Error: {avg_mse}')