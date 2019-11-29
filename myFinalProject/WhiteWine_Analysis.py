# Import Required Libraries# Begin
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Import Requiwhite Libraries# End
np.warnings.filterwarnings('ignore')
# Load The white Wine Dataset CSV File ([1599 rows x 12 columns])
whiteWineDataset = pd.read_csv('Datasets\winequality-white.csv', sep=';', encoding='utf-8')
# Describe the dataset to get minimum and maximum score for quality index
print(whiteWineDataset.describe())
whiteWineDataset.describe().to_csv('Datasets\whiteDataset-Description.csv')
# quality range from 3 up to 8
# find correlation between the dataset features and save them into csv file
whiteWineCorr = whiteWineDataset.corr(method='pearson')
print('********white Wine Correlation', whiteWineCorr)
whiteWineCorr.to_csv('Datasets\correlation-winequality-white.csv')
# Plot the Correlation view
plt.matshow(whiteWineDataset.corr())
plt.xticks(range(len(whiteWineDataset.columns)), whiteWineDataset.columns, rotation=45 )
plt.yticks(range(len(whiteWineDataset.columns)), whiteWineDataset.columns, rotation=0 )
plt.colorbar()
plt.tight_layout()
#plt.savefig('Images\correlation-winequality-white.png')
plt.show()
plt.clf()
sns.set(style="ticks", color_codes=True)
plt.figure(figsize=(10, 10))
sns.pairplot(whiteWineDataset, hue='quality')
plt.savefig('Images\whiteWineQuality.png')
#plt.show()
'''
based on the above  Correlation Insights, we have the below strong correlated features with white wine Quality 

1: alcohol 0.43
2: density -0.30
'''
# Apply ML Model -- Logistic regression on quality feature based on the rest of physical tests (features)
plt.clf()
features = whiteWineDataset.drop('quality', axis=1)
target = whiteWineDataset['quality']
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.40, random_state=42)
lr = LogisticRegression()
lr.fit(x_train, y_train)
# Printing the  difference between real and predicted quality feature
predicted_values = lr.predict(x_test)
data = []
for (real, predicted) in list(zip(y_test, predicted_values)):
   data.append([real, predicted, real-predicted])
df1 = pd.DataFrame(data, columns=['real', 'predicted', 'diff'])
df1.to_csv('Datasets\V1 Differences Distribution.csv')
sns.distplot(df1['diff'], rug=True, hist=True)
plt.tight_layout()
plt.savefig('Images\Distribution for Quality Differences(real-predicted).png')
plt.xlabel('Distribution for Quality Differences(real-predicted)')
plt.show()
plt.clf()
# we are going to increase the Alcohol by 1% from mean
whiteWineDataset['alcoholplus']=round(((whiteWineDataset['alcohol'].mean())/100), 2)+whiteWineDataset['alcohol']
print(whiteWineDataset)
## by using alcoholplus instead of alcohol
# Apply ML Model -- Logistic regression on quality feature based on the rest of physical tests (features)
plt.clf()
features = whiteWineDataset.drop('alcohol', axis=1)
target = whiteWineDataset['quality']
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.40, random_state=42)
lr = LogisticRegression()
lr.fit(x_train, y_train)
# Printing the  difference between real and predicted quality feature
predicted_values = lr.predict(x_test)
data = []
for (real, predicted) in list(zip(y_test, predicted_values)):
   data.append([real, predicted, real-predicted])
df1 = pd.DataFrame(data, columns=['real', 'predicted', 'diff'])
df1.to_csv('Datasets\V2 Differences Distribution.csv')
sns.distplot(df1['diff'], rug=True, hist=True)
plt.tight_layout()
plt.xlabel('V2 Distribution for Quality Differences(real-predicted)')
plt.savefig('Images\V2 Distribution for Quality Differences(real-predicted).png')
plt.xlabel('V2 Distribution for Quality Differences(real-predicted)')
plt.show()
plt.clf()