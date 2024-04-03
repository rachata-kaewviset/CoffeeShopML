import pandas as pd

df = pd.read_csv('Project.csv')

grouped_df = df.groupby(['transaction_date','store_location','unit_price','product_category','product_type','Size','Day Name'])['transaction_qty'].sum().reset_index()
grouped_df.to_csv('grouped_data.csv', index=False)
