install.packages("magrittr")
library(magrittr)
library(mice)
library(caTools)
library(rpart)
library(rpart.plot)
library(tidyverse)
train=read.csv("train.csv")
str(train)
summary(train)

cleantrain=train
cleantrain[cleantrain$PassengerId == ""] = 0
cleantrain[cleantrain$Transported == ""] = 0
str(cleantrain)

finaltrain = cleantrain
finaltrain$VIP=as.factor(finaltrain$VIP)
class(finaltrain$VIP)
finaltrain$CryoSleep=as.factor(finaltrain$CryoSleep)
class(finaltrain$CryoSleep)
finaltrain$Transported=as.factor(finaltrain$Transported)
class(finaltrain$Transported)
endtrain = mice(finaltrain,m=5,method=c("","","polyreg","","","pmm","polyreg","pmm","pmm","pmm","pmm","pmm","","logreg"),maxit=20)

summary(finaltrain$Age)

endtrain$imp$Age
cleantrain=complete(endtrain,1)
summary(cleantrain)


#Checking for the duplicated data
#dup1=sum(duplicated(cleantrain))
#table(dup1)


set.seed(88)
split=sample.split(cleantrain$Transported,SplitRatio = 0.75)
split


# Create training and testing sets
transportedTrain=subset(cleantrain,split==TRUE)
transportedTest=subset(cleantrain,split==FALSE)

# Logistic Regression Model
trainLog=glm(Transported~RoomService+Spa+FoodCourt+HomePlanet
             +Age+Destination+VIP+CryoSleep+VRDeck,
               data =transportedTrain ,family = binomial)

summary(trainLog)

# Decision tree Model
#trainCART=rpart(Transported~RoomService+Spa+FoodCourt+HomePlanet
#                +Age+Destination+VIP+CryoSleep+VRDeck,method="class",data=transportedTrain,control=rpart.control(minbucket=25))
#prp(trainCART)

# Random forest Model
#randomd=randomForest(Transported~deck+num+side+Spa+FoodCourt+HomePlanet
#                     +Age+Destination+VIP+CryoSleep+VRDeck,data=cleantrain,na.action=na.exclude,ntree = 500)

test=read.csv("test.csv")
str(test)

finaltest = test
finaltest$VIP=as.factor(finaltest$VIP)
class(finaltest$VIP)
finaltest$CryoSleep=as.factor(finaltest$CryoSleep)
class(finaltest$CryoSleep)
endtest = mice(finaltest,m=5,method=c("","","polyreg","","","pmm","polyreg","pmm","pmm","pmm","pmm","pmm",""),maxit=20)

cleantest=complete(endtest,1)

#newfile generation
cleantest$Transported=predict(trainLog,newdata=cleantest,type = "response")
solution=data.frame(PassengerId = cleantest$PassengerId,Transported=cleantest$Transported)
length(cleantest$PassengerId)
length(cleantest$Transported)
submission=read.csv("submission_new.csv")
submission$Transported=cleantest$Transported
submission$PassengerId=cleantest$PassengerId


# Line 78 and 79 are used only for the logistic model
#solution2 <- solution %>%
#  mutate(Transported = factor(case_when(Transported <= 0.500000 ~ 'False', Transported > 0.500000 ~ 'True')))


write.csv(solution2,"GLMMICE.csv",row.names = FALSE)

###################################################################################################################################################
###################################################################################################################################################

#XG Boost Python code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
import warnings
warnings.filterwarnings("ignore")

df_train = pd.read_csv("C:/Users/ajink/OneDrive/Desktop/EA1/Project/XGBoost/train.csv")
df_test = pd.read_csv("C:/Users/ajink/OneDrive/Desktop/EA1/Project/XGBoost/test.csv")

df_train.head()
df_test.head()
df_train.describe()
df_test.describe()

print("Training data Shape:",df_train.shape)
print("Testing data Shape:",df_test.shape)

df_train.isna().sum()
df_test.isna().sum()
df_train.nunique()
df_test.nunique()

billed = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
df_train["Total_billed"] = df_train[billed].sum(axis=1)
df_test["Total_billed"] = df_test[billed].sum(axis=1)

df_train.Total_billed.describe()#the min and max value has huge differences so there are oultiers in all of the bill columns
df_test.Total_billed.describe()#the min and max value has huge differences so there are oultiers in all of the bill columns

# filling the null values of the billed columns
for i in billed:
  df_train[i] = df_train[i].fillna(df_train[i].median())
df_test[i] = df_test[i].fillna(df_train[i].median())

df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())
df_test["Age"] = df_test["Age"].fillna(df_train["Age"].median())

df_train.info()

cols = ["HomePlanet","CryoSleep","Cabin","Destination","VIP"]

for i in cols:
  df_train[i] = df_train[i].fillna(st.mode(df_train[i]))
df_test[i] = df_test[i].fillna(st.mode(df_test[i]))

df_train.isna().sum()
X = df_train.drop(["Transported","Cabin","Total_billed","Name"],axis=1)
y = df_train.Transported

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

test = df_test.drop(labels=["Cabin","Name","Total_billed"],axis=1)

X_train.shape
num_col = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck","Age"]
cat_cols = ["HomePlanet","CryoSleep","Destination","VIP","PassengerId"]

df_train[num_col]=df_train[num_col].astype(dtype="int")
df_test[num_col] = df_test[num_col].astype(dtype="int")

le = LabelEncoder()

for i in cat_cols:
  X_train[i] = le.fit_transform(X_train[i])
X_test[i] = le.fit_transform(X_test[i])
test[i] = le.fit_transform(test[i])

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

X_train_LR = X_train
X_test_LR = X_test
test_LR = test

mm = MinMaxScaler()
X_train_LR[num_col] = mm.fit_transform(X_train[num_col])
X_test_LR[num_col] = mm.transform(X_test[num_col])
test_LR[num_col] = mm.transform(test[num_col])

sc = StandardScaler()
X_train[num_col] = sc.fit_transform(X_train[num_col])
X_test[num_col] = sc.transform(X_test[num_col])
test[num_col] = sc.transform(test[num_col])

from sklearn.linear_model import LogisticRegression

model_LR = LogisticRegression()

model_LR.fit(X_train_LR,y_train)

pred_LR = model_LR.predict(X_test_LR)

pred_LR

pred_train_LR = model_LR.predict(X_train_LR)

print("Training Accuracy:",accuracy_score(y_train,pred_train_LR))
print("Testing Accuracy:",accuracy_score(y_test,pred_LR))

c_m_LR = confusion_matrix(y_test,pred_LR)
plt.figure(figsize=(7,5))
ax = sns.heatmap(c_m_LR,annot=True,fmt="g")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.xaxis.set_ticklabels(["Not Diabetic","Diabetic"])
ax.yaxis.set_ticklabels(["Not Diabetic","Diabetic"])

print(classification_report(y_test,pred_LR))

from sklearn.neighbors import KNeighborsClassifier

model_KNN = KNeighborsClassifier(n_neighbors=13)
model_KNN.fit(X_train,y_train)
pred_KNN= model_KNN.predict(X_test)
pred_KNN
pred_train_KNN = model_KNN.predict(X_train)

print("Training Accuracy:",accuracy_score(y_train,pred_train_KNN))
print("Testing Accuracy:",accuracy_score(y_test,pred_KNN))

c_m_KNN = confusion_matrix(y_test,pred_KNN)
plt.figure(figsize=(7,5))
ax = sns.heatmap(c_m_KNN,annot=True,fmt="g")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.xaxis.set_ticklabels(["Not Diabetic","Diabetic"])
ax.yaxis.set_ticklabels(["Not Diabetic","Diabetic"])

print(classification_report(y_test,pred_KNN))

from sklearn.ensemble import RandomForestClassifier

model_RF = RandomForestClassifier(n_estimators= 2000,
                                  min_samples_split= 5,
                                  min_samples_leaf= 1,
                                  max_features= 'sqrt',
                                  max_depth=10,
                                  bootstrap= True)
model_RF.fit(X_train,y_train)

pred_RF = model_RF.predict(X_test)

pred_train_RF = model_RF.predict(X_train)

print("Training Accuracy:",accuracy_score(y_train,pred_train_RF))
print("Testing Accuracy:",accuracy_score(y_test,pred_RF))

pred_test = model_RF.predict(test)

pred_test = np.where(pred_test==0,False,True)

final_submission = pd.DataFrame({"PassengerID":df_test.PassengerId,"Transported":pred_test})

final_submission.to_csv("C:/Users/ajink/OneDrive/Desktop/EA1/Project/XGBoost/submissionRF.csv",index=False)

final_submission

##################################################################################################################################################
##################################################################################################################################################

#Multinomial naive bayes python code

import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/ajink/OneDrive/Desktop/EA1/Project/XGBoost/train.csv")
df_test = pd.read_csv("C:/Users/ajink/OneDrive/Desktop/EA1/Project/XGBoost/test.csv")
df.shape

df_test.shape
df= df.dropna()

df_test['RoomService'].fillna(value=df['RoomService'].mean(), inplace=True)
df_test['FoodCourt'].fillna(value=df['FoodCourt'].mean(), inplace=True)
df_test['ShoppingMall'].fillna(value=df['ShoppingMall'].mean(), inplace=True)
df_test['Spa'].fillna(value=df['Spa'].mean(), inplace=True)
df_test['VRDeck'].fillna(value=df['VRDeck'].mean(), inplace=True)
df_test['Age'].fillna(value=df['Age'].mean(), inplace=True)

df.shape
features = ['HomePlanet','CryoSleep','Destination','VIP']
features_test = features
columns=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','HomePlanet','CryoSleep','Destination','Age','VIP','PassengerId','Cabin','Name','Age']
columns_test = columns


for col in features:
  s = df[col].unique()

# Create a One Hot Dataframe with 1 row for each unique value
one_hot_df = pd.get_dummies(s, prefix='%s_' % col)
one_hot_df[col] = s

print("Adding One Hot values for %s (the column has %d unique values)" % (col, len(s)))
pre_len = len(df)

# Merge the one hot columns
df = df.merge(one_hot_df, on=[col], how="left")
assert len(df) == pre_len
print(df.shape)


for col in features:
  s = df_test[col].unique()

# Create a One Hot Dataframe with 1 row for each unique value
one_hot_df = pd.get_dummies(s, prefix='%s_' % col)
one_hot_df[col] = s

print("Adding One Hot values for %s (the column has %d unique values)" % (col, len(s)))
pre_len = len(df)

# Merge the one hot columns
df_test = df_test.merge(one_hot_df, on=[col], how="left")
assert len(df) == pre_len
print(df_test.shape)
df1 = df.drop(columns,axis=1)

df_test_1 = df_test.drop(columns,axis=1)
df1['RoomService'] = df['RoomService']
df1['FoodCourt'] = df['FoodCourt']
df1['ShoppingMall'] = df['ShoppingMall']
df1['Spa'] = df['Spa']
df1['VRDeck'] = df['VRDeck']
df1['Age'] = df['Age']
df_test_1['RoomService'] = df_test['RoomService']
df_test_1['FoodCourt'] = df_test['FoodCourt']
df_test_1['ShoppingMall'] = df_test['ShoppingMall']
df_test_1['Spa'] = df_test['Spa']
df_test_1['VRDeck'] = df_test['VRDeck']
df_test_1['Age'] = df_test['Age']
df_test_1
df2 = df1.loc[:, 'Transported']
df2
df1 = df1.drop('Transported',axis=1)
from sklearn.model_selection import train_test_split
RANDOM_STATE = 1234

X_train = df1
y_train = df2

X_test = df_test_1

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

def model(classifier, X_train, X_test):
  classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test)
return y_pred

from sklearn.naive_bayes import MultinomialNB
y_pred = model(MultinomialNB(), X_train, X_test)

print(y_pred)

test_result= pd.DataFrame(y_pred)
test_result.columns = ['Transported']
predict = test_result['Transported']
Id_No = df_test['PassengerId']
submission = pd.DataFrame({'PassengerId': Id_No, 'Transported': predict})
submission['Transported'] = (submission['Transported']>=0.5).astype(int)
submission=submission.replace({0:False, 1:True})
submission.to_csv('C:/Users/ajink/OneDrive/Desktop/EA1/Project/XGBoost/submission5.csv', index=False)


