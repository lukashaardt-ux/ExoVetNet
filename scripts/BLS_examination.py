import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('data/BLSvCatalog_data.csv')

print(df.head())

period_diffs = np.abs(df['cat_period'] - df['period']) / df['cat_period']
epoch_diffs  = np.abs(df['cat_epoch']  - df['epoch'])  / df['cat_epoch']
duration_diffs = np.abs(df['cat_duration'] - df['duration']) / df['cat_duration']


print(f"Mean period difference: {np.mean(period_diffs)}")
print(f"Mean epoch difference: {np.mean(epoch_diffs)}")
print(f"Mean duration difference: {np.mean(duration_diffs)}")

print(f"Median period difference: {np.median(period_diffs)}")
print(f"Median epoch difference: {np.median(epoch_diffs)}")
print(f"Median duration difference: {np.median(duration_diffs)}")

print("within 1%:", (period_diffs < 0.01).mean())
print("within 0.1%:", (period_diffs < 0.001).mean())
print("drifted >5%:", (period_diffs > 0.05).mean())
print("at window edge (>9%):", (period_diffs > 0.09).mean())
print("very large difference (>15%):", (period_diffs > 0.15).mean())
'''
print("within 1%:", (epoch_diffs < 0.01).mean())
print("within 0.1%:", (epoch_diffs < 0.001).mean())
print("drifted >5%:", (epoch_diffs > 0.05).mean())
print("at window edge (>9%):", (epoch_diffs > 0.09).mean())

print("within 1%:", (duration_diffs < 0.01).mean())
print("within 0.1%:", (duration_diffs < 0.001).mean())
print("drifted >5%:", (duration_diffs > 0.05).mean())
print("at window edge (>9%):", (duration_diffs > 0.09).mean())
'''
plt.style.use("ggplot")
plt.hist(period_diffs, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel("Relative period difference (BLS vs catalog)")
plt.ylabel("Count")
plt.show()

good_names = df[period_diffs < 0.01]['name'].to_list()

