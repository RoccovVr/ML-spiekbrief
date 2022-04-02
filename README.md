#---------------------------
#machine-learning

#imports

#required:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split        # Load train/test split

#required, but the needed model varies:
from sklearn import neighbors                               # Load model

#sometimes handy:
from sklearn.model_selection import cross_val_score         # CV
from sklearn.model_selection import GridSearchCV            # grid search

#__________________________________________

#train-test codes

#train test data:
df = sns.load_dataset("iris")

X = df.drop(['species'], axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

#examples of instantiating models: 

#Ex.1 - standard: 
classifier = neighbors.KNeighborsClassifier(n_neighbors=3) #or

#Ex.2 - with grid:
param_grid_RF = {'n_estimators': [2, 4, 8, 16, 32],
                 'max_depth': [2, 3, 4, 5, 6]}
grid = GridSearchCV(RandomForestClassifier(), param_grid_RF, cv=6)

#Ex.3 - with pipe:
pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC())]) #StandardScaler() is more usual

param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
              'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)

#training:
classifier.fit(X_train, y_train) 

#gives test scores:
classifier.score(X_test, y_test) #or
cv_scores = cross_val_score(classifier, X, y, cv=5)       #does CV on whole dataset X,y and gives scores


# Print best parameters and corresponding score
print(grid.best_params_)
print(grid.best_score_)

#-----------------------------------

#dimreduction:
#if high-dim, then use kernelPCA, otherwise PCA

#PCA example:
from sklearn.decomposition import PCA
pipe = Pipeline([(‘scaler’, StandardScaler()),
 (‘pca’, PCA(n_components=3)), #here PCs remain
 (‘clf’, RandomForestClassifier())])
pipe.fit(X_train,y_train)

#-----------------------------------
#Ensemble models

# Create the sub models
from sklearn.ensemble import VotingClassifier
estimators = []
logModel = LogisticRegression()
estimators.append(('logistic', logModel))
cartModel = DecisionTreeClassifier()
estimators.append(('cart', cartModel))
svmModel = SVC()
estimators.append(('svm', svmModel))
# create the ensemble model
ensemble = VotingClassifier(estimators)

#-----------------------------------------
#Preprocessing:

#improves comp time:
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y_transform = le.fit_transform(y) #when y is a target variable with names

#be aware of missing values, remove them or inpute
#checking for nb of NaN values per row:
df.isnull().sum().to_frame()

#check if dataset is unbalanced:
df.response.value_counts() #response is name of target var
#if so, use stratify in the train test split
