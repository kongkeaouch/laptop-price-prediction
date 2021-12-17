from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv('laptop.csv')
df.head()
df.columns
df = df[['Company', 'Type', 'Size', 'Resolution',
         'CPU', 'RAM', 'Storage', 'GPU', 'OS', 'Weight', 'Price']]
df.head()
df.isnull().sum()
df.duplicated().sum()
df.info()
catvars = df.select_dtypes(include=['object']).columns
numvars = df.select_dtypes(
    include=['int32', 'int64', 'float32', 'float64']).columns
catvars, numvars


def uniquevals(col):
    print(f"Details {col} is : {df[col].unique()}")


def valuecounts(col):
    print(
        f"Valuecounts {col} is : {df[col].value_counts()}")


for col in df.columns:
    uniquevals(col)
    print('-'*75)
df['RAM'] = df['RAM'].str.replace('GB', '')
df['Weight'] = df['Weight'].str.replace('kg', '')
df['RAM'] = df['RAM'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')
df.head()
df.info()
sn.distplot(df['Price'], color='red')


def drawplot(col):
    plt.figure(figsize=(15, 7))
    sn.countplot(df[col], palette='plasma')
    plt.xticks(rotation='vertical')


toview = ['Company', 'Type', 'RAM', 'OS']
for col in toview:
    drawplot(col)
plt.figure(figsize=(15, 7))
sn.barplot(x=df['Company'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()
sn.countplot(df['Type'], palette='autumn')
plt.xticks(rotation='vertical')
sn.barplot(x=df['Type'], y=df['Price'])
plt.xticks(rotation='vertical')
sn.scatterplot(x=df['Size'], y=df['Price'])
df['Resolution'].value_counts()
df['TouchScreen'] = df['Resolution'].apply(
    lambda element: 1 if'Touchscreen' in element else 0)
df.head()
df.sample(5)

sn.countplot(df['TouchScreen'], palette='plasma')
sn.barplot(x=df['TouchScreen'], y=df['Price'])
plt.xticks(rotation='vertical')
df['IPS'] = df['Resolution'].apply(
    lambda element: 1 if'IPS' in element else 0)
df.sample(5)
sn.countplot(df['IPS'], palette='plasma')
sn.barplot(x=df['TouchScreen'], y=df['Price'])
plt.xticks(rotation='vertical')
splitdf = df['Resolution'].str.split('x', n=1, expand=True)
splitdf.head()
splitdf = df['Resolution'].str.split('x', n=1, expand=True)
df['X_res'] = splitdf[0]
df['Y_res'] = splitdf[1]
df.head()
df['X_res'] = df['X_res'].str.replace(',', '').str.findall(
    '(\\d+\\.?\\d+)').apply(lambda x: x[0])
df.head()
df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')
df.info()
plt.figure(figsize=(15, 7))
sn.heatmap(df.corr(), annot=True, cmap='plasma')
df.corr()['Price']
df.head()
df.corr()['Price']
df.drop(columns=['Resolution', 'Size', 'X_res', 'Y_res'], inplace=True)
df.head()
df['CPU'].value_counts()
df['CPU_name'] = df['CPU'].apply(lambda text: ' '.join(text.split()[:3]))
df.head()


def processortype(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    elif text.split()[0] == 'Intel':
        return'Other Intel Processor'
    else:
        return'AMD Processor'


df['CPU_name'] = df['CPU_name'].apply(lambda text: processortype(text))
df.head()
sn.countplot(df['CPU_name'], palette='plasma')
plt.xticks(rotation='vertical')
sn.barplot(df['CPU_name'], df['Price'])
plt.xticks(rotation='vertical')
df.drop(columns=['CPU'], inplace=True)
df.head()
sn.countplot(df['RAM'], palette='autumn')
sn.barplot(df['RAM'], df['Price'])
df['Storage'].iloc[:1][0]
df['Storage'].value_counts()
df['Storage'] = df['Storage'].astype(str).replace('\\.0', '', regex=True)
df['Storage'] = df['Storage'].str.replace('GB', '')
df['Storage'] = df['Storage'].str.replace('TB', '000')
newdf = df['Storage'].str.split('+', n=1, expand=True)
newdf
df['first'] = newdf[0]
df['first'] = df['first'].str.strip()
df.head()


def applychanges(value):
    df['Layer1'+value] = df['first'].apply(lambda x: 1 if value in x else 0)


listtoapply = ['HDD', 'SSD', 'Hybrid', 'FlashStorage']
for value in listtoapply:
    applychanges(value)
df.head()
df['first'] = df['first'].str.replace('\\D', '')
df['first'].value_counts()
df['Second'] = newdf[1]
df.head()


def applychanges1(value):
    df['Layer2'+value] = df['Second'].apply(lambda x: 1 if value in x else 0)


listtoapply1 = ['HDD', 'SSD', 'Hybrid', 'FlashStorage']
df['Second'] = df['Second'].fillna('0')
for value in listtoapply1:
    applychanges1(value)
df['Second'] = df['Second'].str.replace('\\D', '')
df['Second'].value_counts()
df['first'] = df['first'].astype('int')
df['Second'] = df['Second'].astype('int')
df.head()
df['HDD'] = df['first']*df['Layer1HDD']+df['Second']*df['Layer2HDD']
df['SSD'] = df['first']*df['Layer1SSD']+df['Second']*df['Layer2SSD']
df['Hybrid'] = df['first']*df['Layer1Hybrid']+df['Second']*df['Layer2Hybrid']
df['Flash_Storage'] = df['first']*df['Layer1FlashStorage'] + \
    df['Second']*df['Layer2FlashStorage']
df.drop(columns=['first', 'Second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1FlashStorage',
                 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid', 'Layer2FlashStorage'], inplace=True)
df.sample(5)
df.drop(columns=['Storage'], inplace=True)
df.sample(5)
df.corr()['Price']
df.columns
df.drop(columns=['Hybrid', 'Flash_Storage'], inplace=True)
df.head()
df['GPU'].value_counts()
a = df['GPU'].iloc[1]
print(a.split()[0])
df['GPU brand'] = df['GPU'].apply(lambda x: x.split()[0])
sn.countplot(df['GPU brand'], palette='plasma')
df = df[df['GPU brand'] != 'ARM']
sn.countplot(df['GPU brand'], palette='plasma')
sn.barplot(df['GPU brand'], df['Price'], estimator=np.median)
df = df.drop(columns=['GPU'])
df.head()
df['OS'].value_counts()
sn.barplot(df['OS'], df['Price'])
plt.xticks(rotation='vertical')
plt.show()
df['OS'].unique()


def setcategory(text):
    if text == 'Windows 10' or text == 'Windows 7' or text == 'Windows 10 S':
        return'Windows'
    elif text == 'Mac OS X' or text == 'macOS':
        return'Mac'
    else:
        return'Other'


df['OS'] = df['OS'].apply(lambda x: setcategory(x))
df.head()
df.sample(5)
sn.countplot(df['OS'], palette='plasma')
sn.barplot(x=df['OS'], y=df['Price'])
plt.xticks(rotation='vertical')
sn.distplot(df['Weight'])
sn.scatterplot(df['Weight'], df['Price'])
sn.distplot(df['Price'])
sn.distplot(np.log(df['Price']))
df.corr()['Price']
plt.figure(figsize=(10, 5))
sn.heatmap(df.corr(), annot=True, cmap='plasma')
test = np.log(df['Price'])
train = df.drop(['Price'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    train, test, test_size=0.15, random_state=2)
X_train.shape, X_test.shape
mapper = {i: value for(i, value) in enumerate(X_train.columns)}
mapper
step1 = ColumnTransformer(transformers=[('col_tnf', OneHotEncoder(
    sparse=False, drop='first'), [0, 1, 3, 8, 11])], remainder='passthrough')
step2 = LinearRegression()
pipe = Pipeline([('step1', step1), ('step2', step2)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 score', metrics.r2_score(y_test, y_pred))
print('MAE', metrics.mean_absolute_error(y_test, y_pred))
np.exp(0.21)

step1 = ColumnTransformer(transformers=[('col_tnf', OneHotEncoder(
    sparse=False, drop='first'), [0, 1, 3, 8, 11])], remainder='passthrough')
step2 = Ridge(alpha=10)
pipe = Pipeline([('step1', step1), ('step2', step2)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 score', metrics.r2_score(y_test, y_pred))
print('MAE', metrics.mean_absolute_error(y_test, y_pred))

step1 = ColumnTransformer(transformers=[('col_tnf', OneHotEncoder(
    sparse=False, drop='first'), [0, 1, 3, 8, 11])], remainder='passthrough')
step2 = Lasso(alpha=0.001)
pipe = Pipeline([('step1', step1), ('step2', step2)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 score', metrics.r2_score(y_test, y_pred))
print('MAE', metrics.mean_absolute_error(y_test, y_pred))

step1 = ColumnTransformer(transformers=[('col_tnf', OneHotEncoder(
    sparse=False, drop='first'), [0, 1, 3, 8, 11])], remainder='passthrough')
step2 = DecisionTreeRegressor(max_depth=8)
pipe = Pipeline([('step1', step1), ('step2', step2)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 score', metrics.r2_score(y_test, y_pred))
print('MAE', metrics.mean_absolute_error(y_test, y_pred))

step1 = ColumnTransformer(transformers=[('col_tnf', OneHotEncoder(
    sparse=False, drop='first'), [0, 1, 3, 8, 11])], remainder='passthrough')
step2 = RandomForestRegressor(
    n_estimators=100, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15)
pipe = Pipeline([('step1', step1), ('step2', step2)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 score', metrics.r2_score(y_test, y_pred))
print('MAE', metrics.mean_absolute_error(y_test, y_pred))

pickle.dump(df, open('df.pkl', 'wb'))
pickle.dump(pipe, open('pipe.pkl', 'wb'))
train.head()
train.to_csv('traineddata.csv', index=None)
indexlist = [0, 1, 3, 8, 11]
transformlist = []
for (key, value) in mapper.items():
    if key in indexlist:
        transformlist.append(value)
transformlist
train = pd.get_dummies(train, columns=transformlist, drop_first=True)
train.head()
X_train, X_test, y_train, y_test = train_test_split(
    train, test, test_size=0.15, random_state=2)
X_train.shape, X_test.shape
reg = DecisionTreeRegressor(random_state=0)
reg.fit(X_train, y_train)
plt.figure(figsize=(16, 9))
tree.plot_tree(reg, filled=True, feature_names=train.columns)
path = reg.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
alphalist = []

for alpha in ccp_alphas:
    reg = DecisionTreeRegressor(random_state=0, ccp_alpha=alpha)
    reg.fit(X_train, y_train)
    alphalist.append(reg)
train_score = [reg.score(X_train, y_train)for reg in alphalist]
test_score = [reg.score(X_test, y_test)for reg in alphalist]
plt.xlabel('ccp alpha')
plt.ylabel('Accuracy')
plt.plot(ccp_alphas, train_score, marker='o',
         label='training', color='magenta')
plt.plot(ccp_alphas, test_score, marker='+', label='testing',
         color='red', drawstyle='steps-post')
plt.legend()
plt.show()
reg = DecisionTreeRegressor(random_state=0, ccp_alpha=0.0085)
reg.fit(X_train, y_train)
plt.figure(figsize=(16, 9))
tree.plot_tree(reg, filled=True, feature_names=train.columns)
params = {'RandomForest': {'model': RandomForestRegressor(), 'params': {'n_estimators': [int(x)for x in np.linspace(100, 1200, 10)], 'criterion': ['mse', 'mae'], 'max_depth': [int(x)for x in np.linspace(1, 30, 5)], 'max_features': ['auto', 'sqrt', 'log2'], 'ccp_alpha': [x for x in np.linspace(0.0025, 0.0125, 5)], 'min_samples_split': [2, 5, 10, 14], 'min_samples_leaf': [
    2, 5, 10, 14]}}, 'Decision Tree': {'model': DecisionTreeRegressor(), 'params': {'criterion': ['mse', 'mae'], 'max_depth': [int(x)for x in np.linspace(1, 30, 5)], 'max_features': ['auto', 'sqrt', 'log2'], 'ccp_alpha': [x for x in np.linspace(0.0025, 0.0125, 5)], 'min_samples_split': [2, 5, 10, 14], 'min_samples_leaf': [2, 5, 10, 14]}}}
scores = []

for (modelname, mp) in params.items():
    clf = RandomizedSearchCV(mp['model'], param_distributions=mp['params'],
                             cv=5, n_iter=10, scoring='neg_mean_squared_error', verbose=2)
    clf.fit(X_train, y_train)
    scores.append({'model_name': modelname, 'best_score': clf.best_score_,
                   'best_estimator': clf.best_estimator_})
scores_df = pd.DataFrame(
    scores, columns=['model_name', 'best_score', 'best_estimator'])
scores_df
scores
rf = RandomForestRegressor(ccp_alpha=0.0025, max_depth=22,
                           min_samples_leaf=14, min_samples_split=5, n_estimators=1200)
rf.fit(X_train, y_train)
ypred = rf.predict(X_test)
print(metrics.r2_score(y_test, y_pred))
predicted = []
testtrain = np.array(train)

for i in range(len(testtrain)):
    predicted.append(rf.predict([testtrain[i]]))
predicted
ans = [np.exp(predicted[i][0])for i in range(len(predicted))]
df['Predicted Price'] = np.array(ans)
df
sn.distplot(df['Price'], hist=False, color='orange', label='Actual')
sn.distplot(df['Predicted Price'], hist=False, color='blue', label='Predicted')
plt.legend()
plt.show()
rf1 = RandomForestRegressor(
    n_estimators=100, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15)
rf1.fit(X_train, y_train)
print(f"R2 score : {metrics.r2_score(y_test,rf1.predict(X_test))}")
predicted = []
testtrain = np.array(train)

for i in range(len(testtrain)):
    predicted.append(rf1.predict([testtrain[i]]))
predicted
ans = [np.exp(predicted[i][0])for i in range(len(predicted))]
data = df.copy()
data['Predicted Price'] = np.array(ans)
data
sn.distplot(data['Price'], hist=False, color='orange', label='Actual')
sn.distplot(data['Predicted Price'], hist=False,
            color='blue', label='Predicted')
plt.legend()
plt.show()

file = open('laptoppricepredictor.pkl', 'wb')
pickle.dump(rf1, file)
file.close()
X_train.iloc[0]
