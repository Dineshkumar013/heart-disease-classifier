 # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))# %% [code] {"id":"RUQJigZ-TgUI"}
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import neighbors,datasets,preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# %% [code] {"id":"g728MKzVTkYJ"}
# getting the dataset
df = pd.read_csv("../input/MLChallenge-2/final.csv")

# %% [code] {"id":"jS5PCjsrTrFM","outputId":"59d4e4c7-4614-47ad-9d21-4855b5e42b82"}
print(df)

# %% [code] {"id":"Rmu-TFGpTtgv","outputId":"8e03bbe5-1935-4186-d9f0-fab9b03a4419"}
df.target.value_counts()
countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))
# %% [code] {"id":"22MFJ0y0UjVo"}
X_data=df.drop(['target'],axis=1)
Y=df.target.values
#visualizing the data 
sns.countplot(x='sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()

# %% [code] {"id":"vToOYeuEUpr4","outputId":"652f6a11-2237-4fb5-f721-9c4da3af13e4"}
print(X_data)
print(Y)



# %% [code] {"id":"PEg_DKBSVLlC"}
#x = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data)).values

# %% [code] {"id":"wP515LF7VMv1"}
#train and test the dataset
x_train, x_test, y_train, y_test = train_test_split(X_data,Y,test_size = 0.2,random_state=0)

# %% [code] {"id":"v6HWgqRIVS7N","outputId":"78e3ccf8-c6eb-4332-edc0-812abaf8d690"}
print(Y.shape, y_train.shape, y_test.shape)

# %% [code] {"id":"eAv6qiJyVXM3"}
x_train, x_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.2, stratify=Y)

# %% [code] {"id":"9a-mSt4fVe6g","outputId":"ae1c2cf7-b75f-4f7c-d020-68cba2944a1e"}
print(Y.mean(), y_train.mean(), y_test.mean())

# %% [code] {"id":"-Nq1q2c_VmpH","outputId":"e994027f-8d51-4ec8-d4ea-5a227110e335"}
print(x_train.mean(), x_test.mean(), X_data.mean())

# %% [code] {"id":"a3Rf-R-YVq0o","outputId":"d063afbc-02a1-42c0-e0e5-805f1aea281f"}
# import Logistic Regression from sklearn
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression() # loading the logistic regression model to the variable "classifier"
classifier.fit(x_train, y_train)

# %% [code] {"id":"hixLzB8lVyJO","outputId":"7d814306-4ed6-445b-9107-2d6559b50829"}
#prediction on training_data
prediction_on_training_data = classifier.predict(x_train)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)
print('Accuracy on training data : ', accuracy_on_training_data)

# %% [code] {"id":"zlk8O2UtV3JK","outputId":"2f0d7ec5-adf8-4fe6-ceaf-11877cb361ec"}
# prediction on test_data
prediction_on_test_data = classifier.predict(x_test)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)
print('Accuracy on testing data : ', accuracy_on_test_data)

# %% [code] {"id":"e-jeGh-D-EFa","outputId":"b110bbcc-15df-44de-e97c-52e38ab30d3d"}
input_data = pd.read_csv("../input/MLChallenge-2/Test.csv")
input_data_as_numpy_array = np.array(input_data)
from sklearn.preprocessing import StandardScaler
x_scalar = StandardScaler()
# reshape the array as we are predicting the output for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#prediction
prediction = x_scalar.fit_transform(input_data)
y_pred = classifier.predict(prediction)

  
print(y_pred)

