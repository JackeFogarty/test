import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"c:\Users\JFOGARTY\Downloads\GSC_Data.csv"
data = pd.read_csv(file_path)

file_path_cd = r"c:\Users\JFOGARTY\Downloads\ptc_gsc_compare.csv"
comparitive_data = pd.read_csv(file_path_cd)



#   Data cleaning and preperation, find missing values per column. Drop missing values and prices over $100 aliased as clean_data
missing_values = data.isnull().sum()

for col in data:
    if data[col].isnull().any():
        print(col, data[col].isnull().sum())


price_data = ['AVG_PRICE', 'MIN_PRICE', 'MAX_PRICE']
data_no_na = data.dropna(axis=0, subset=price_data)


clean_data = data_no_na[data_no_na['AVG_PRICE'] <= 50]


#   Comparitive price ranges
comparitive_data['Time'] = comparitive_data['Time'].str.replace('Week Ending ', '')
comparitive_data['Time'] = pd.to_datetime(comparitive_data['Time'], format = '%m-%d-%y')
comparitive_data_time = comparitive_data[(comparitive_data['Time'] >= '2023-11-01') & (comparitive_data['Time'] <= '2024-02-29')]

products_to_filter = [
    'CORONA EXTRA BOTTLE 6 CT 12 OZ', 
    'CORONA EXTRA BOTTLE 24 CT 12 OZ', 
    'CORONA EXTRA BOTTLE 18 CT 12 OZ', 
    'CORONA EXTRA BOTTLE 12 CT 12 OZ', 
    'CORONA EXTRA BOTTLE 1 CT 12 OZ', 
    'CORONA EXTRA BOTTLE 1 CT 24 OZ', 
    'CORONA EXTRA BOTTLE 1 CT 7 OZ',
    'CORONA EXTRA CAN 1 CT 16 OZ',
    'CORONA EXTRA CAN 1 CT 24 OZ ',
    'CORONA EXTRA CAN 3 CT 24 OZ',        
    'CORONA EXTRA CAN 4 CT 16 OZ',
    'CORONA EXTRA CAN 6 CT 12 OZ',
    'CORONA EXTRA CAN 12 CT 12 OZ',
    'CORONA EXTRA CAN 18 CT 12 OZ',
    'CORONA EXTRA CAN 24 CT 12 OZ'
]

filtered_products = comparitive_data[comparitive_data['Product'].isin(products_to_filter)]

real_product_prices = filtered_products.groupby('Product')['Weighted Average Base Price Per Unit'].agg(['min', 'max', 'mean', 'median'])

# Comparing GSC to ranges

gsc_product_grouped = clean_data.groupby(['PACK_TYPE', 'PACK_SIZE', 'SIZE'])['AVG_PRICE'].agg(['min', 'max', 'mean', 'count', 'median'])

print(real_product_prices)
print(gsc_product_grouped)

pack_stds = clean_data.groupby('PACK_SIZE')['AVG_PRICE'].std().round(2)

grouped = data.groupby('PACK_SIZE')['AVG_PRICE']
std_dev = grouped.transform('std')
mean_price = grouped.transform('mean')

data['is_outlier'] = abs(data['AVG_PRICE'] - mean_price) > 2 * std_dev

data_cleaned = data[~data['is_outlier']].drop(columns=['is_outlier'])

# Finding standard deviation of different data sets

data_cleaned_std = data_cleaned.groupby('PACK_SIZE')['AVG_PRICE'].std().round(2)
clean_data_std = clean_data.groupby('PACK_SIZE')['AVG_PRICE'].std().round(2)

pc_std = comparitive_data.groupby('Product')['Weighted Average Base Price Per Unit'].std().round(2)

print(pc_std)
print(data_cleaned_std)
print(clean_data_std)

