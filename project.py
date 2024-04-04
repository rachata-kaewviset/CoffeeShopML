import pandas as pd
import numpy as np
from LinearRegression import LinearRegression

df = pd.read_csv('grouped_data.csv')
df = df.drop(['transaction_date'], axis=1)
df = df.drop(['store_location'], axis=1)
df['transaction_qty'] = (df['transaction_qty'] - df['transaction_qty'].min()) / (df['transaction_qty'].max() - df['transaction_qty'].min())
df['unit_price'] = (df['unit_price'] - df['unit_price'].min()) / (df['unit_price'].max() - df['unit_price'].min())

df.replace({'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}, inplace=True)
df.replace({'Bakery': 0, 'Branded': 1, 'Coffee': 2, 'Coffee beans': 3, 'Drinking Chocolate': 4, 'Flavours': 5, 'Loose Tea': 6, 'Packaged Chocolate': 7, 'Tea': 8}, inplace=True)
df.replace({'Biscotti': 0, 'Pastry': 1, 'Scone': 2, 'Clothing': 3, 'Housewares': 4, 'Barista Espresso': 5, 'Drip coffee': 6, 'Gourmet brewed coffee': 7, 'Organic brewed coffee': 8, 'Premium brewed coffee': 9, 'Espresso Beans': 10, 'Gourmet Beans': 11, 'Green beans': 12, 'House blend Beans': 13, 'Organic Beans': 14, 'Premium Beans': 15, 'Hot chocolate': 16, 'Regular syrup': 17, 'Sugar free syrup': 18, 'Black tea': 19, 'Chai tea': 20, 'Green tea': 21, 'Herbal tea': 22, 'Drinking Chocolate': 23, 'Organic Chocolate': 24, 'Brewed Black tea': 25, 'Brewed Chai tea': 26, 'Brewed Green tea': 27, 'Brewed herbal tea': 28}, inplace=True)
df.replace({'Large': 0, 'Not Defined': 1, 'Regular': 1, 'Small': 2}, inplace=True)

X = df.drop(['transaction_qty'], axis=1)
y = df['transaction_qty']

def k_fold_split(X, y, n_splits=10, shuffle=True):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_size = len(X) // n_splits
    for i in range(0, len(X), fold_size):
        test_indices = indices[i:i+fold_size]
        train_indices = np.concatenate((indices[:i], indices[i+fold_size:]))
        yield X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

model = LinearRegression()

mae_scores = []
kf = k_fold_split(X, y)
for X_train, X_test, y_train, y_test in kf:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred))
    mae_scores.append(mae)

avg_mae = sum(mae_scores) / len(mae_scores)

print(f'Average absolute Squared Error: {avg_mae}')
