import pandas as pd
import matplotlib.pyplot as plt

# Read in the data
df = pd.read_csv('Mean Time (s) Mean Distance (-).txt', sep=',')

# keep first two digits
df['Mean Time (s)'] = df['Mean Time (s)'].str[:3]
df['Mean Distance (-)'] = df['Mean Distance (-)'].str[:3]

# convert to numeric
df['Mean Time (s)'] = pd.to_numeric(df['Mean Time (s)'])
df['Mean Distance (-)'] = pd.to_numeric(df['Mean Distance (-)'])

print(df)
# make scatter plot
plt.scatter(df['Mean Time (s)'], df['Mean Distance (-)'])
plt.xlabel('Mean Time (s)')
plt.ylabel('Mean Distance (-)')
plt.title('Mean Time (s) vs Mean Distance (-)')
plt.show()
