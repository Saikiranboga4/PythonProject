import numpy as np
import pandas as pd

data = {
    'month': ['January', 'February', 'March', 'April', 'May', 'June'],
    'sales': [100, 150, 200, 250, 300, 350],
    'expenses': [80, 120, 160, 200, 240, 280],
    'profit': [20, 30, 40, 50, 60, 70],
    'unit': [10, 15, 20, 25, 30, 35],
    'region': ['North', 'South', 'East', 'West', 'Central', 'International']
}

df = pd.DataFrame(data)
print(df)

df['avg_price'] = df['sales'] / df['unit']
df['top_month'] = np.where(df['sales'] > 200, True, False)
df['avg_price'] = np.round(df['avg_price'], 2)
high = df[df['sales'] > 200]
north = df[df['region'] == 'North']
hot = df[(df['sales'] > 200) & (df['region'] == 'North')]
print(high[['month', 'sales']])

summary = df.groupby('region').agg(
    total_sales = ('sales', 'sum'),
    avg_units = ('unit', 'mean'),
    months = ('month', 'count')
).reset_index()
print(summary)

sales = df['sales'].values

print('total :', np.sum(sales))
print('Mean :', np.mean(sales).round(2))
print('Standard Deviation :', np.std(sales).round(2))   
print('Max :', np.max(sales))
print('Correlation :', np.corrcoef(sales, df['unit'].values)[0, 1].round(3))

df.to_csv('sales_analysis.csv', index=False)

df.to_excel('sales_analysis.xlsx', index=False)

summary.to_csv('region_summary.csv', index=False)

print('Files saved successfully.')
