import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# import numpy as np
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
melbourne_file_path = '../ressources/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.columns)
# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)


# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)
y = melbourne_data.Price
melbourne_features = ['Rooms','Distance','Landsize','Bathroom', 'Landsize','Car', 'Lattitude', 'Longtitude','Propertycount','YearBuilt']
X = melbourne_data[melbourne_features]
# print(X.isnull().sum())
print(X.head(3))


sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.distplot(y, bins=30)
#

sns.histplot(y, kde=True, stat="density")
plt.show()


corr = X.astype('int64').corr()
# print(corr)
sns.heatmap(corr.round(2),annot=True)
sns.set(rc={'figure.figsize':(170.7,140.27)})
plt.show()
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=10)


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)




# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(x_train, y_train)
print("\nScore is")
# print(melbourne_model.predict(x_test))
print(melbourne_model.score(x_train, y_train))
# 0.57
### overfitting



from sklearn.linear_model import LinearRegression
model = LinearRegression()
print("\n linear reg")
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
#0.56
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
print("\n forest reg")
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
#0.74
