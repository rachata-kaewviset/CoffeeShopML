import pandas as pd
import numpy as np
from LinearRegression import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('grouped_data.csv')
df = df.drop(['transaction_date'],axis=1)
df.replace('Monday', 0, inplace=True)
df.replace('Tuesday', 1, inplace=True)
df.replace('Wednesday', 2, inplace=True)
df.replace('Thursday', 3, inplace=True)
df.replace('Friday', -1, inplace=True)
df.replace('Saturday', -2, inplace=True)
df.replace('Sunday', -3, inplace=True)

df.replace('Astoria', -1, inplace=True)
df.replace("Hell's Kitchen", 0, inplace=True)
df.replace('Lower Manhattan', 1, inplace=True)

df.replace('Bakery', 0, inplace=True)
df.replace('Branded', 1, inplace=True)
df.replace('Coffee', 2, inplace=True)
df.replace('Coffee beans', 3, inplace=True)
df.replace('Drinking Chocolate', 4, inplace=True)
df.replace('Flavours', -1, inplace=True)
df.replace('Loose Tea', -2, inplace=True)
df.replace('Packaged Chocolate', -3, inplace=True)
df.replace('Tea', -4, inplace=True)

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
df.replace('Hot chocolate', -1, inplace=True)
df.replace('House blend Beans', -2, inplace=True)
df.replace('Housewares', -3, inplace=True)
df.replace('Organic Beans', -4, inplace=True)
df.replace('Organic Chocolate', -5, inplace=True)
df.replace('Organic brewed coffee', -6, inplace=True)
df.replace('Pastry', -7, inplace=True)
df.replace('Premium Beans', -8, inplace=True)
df.replace('Premium brewed coffee', -9, inplace=True)
df.replace('Regular syrup', -10, inplace=True)
df.replace('Scone', -11, inplace=True)
df.replace('Sugar free syrup', -12, inplace=True)
df.replace('Brewed Chai tea', -13, inplace=True)
df.replace('Chai tea', -14, inplace=True)

df.replace('Large', -1, inplace=True)
df.replace('Not Defined', 0, inplace=True)
df.replace('Regular', 1, inplace=True)
df.replace('Small', 2, inplace=True)

X = df.drop(['transaction_qty'], axis=1)
y = df['transaction_qty']

kf = KFold(n_splits=10, shuffle=True)
model = LinearRegression(alpha=0.1, regularization='l2')

mse_scores = []
r2_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_scores.append(mse)
    r2_scores.append(r2)


avg_mse = sum(mse_scores) / len(mse_scores)
avg_r2 = sum(r2_scores) / len(r2_scores)

print(f'Average Mean Squared Error: {avg_mse}')
print(f'Average R-squared: {avg_r2}')
