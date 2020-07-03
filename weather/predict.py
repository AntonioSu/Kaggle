import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def data_loader():
    csv_data=pd.read_csv('data.csv',encoding='gbk')
    return csv_data

def weather_condition(train):
    newDF=train['天气状况'].str.split('/',expand=True)  # 测试包含哪些风力
    newDF.columns = ['天气状况白天', '天气状况晚上']
    global num_label
    num_label=0
    dict={}
    def transform(item):
        global num_label
        if item in dict:
            return dict[item]
        else:
            dict[item]=num_label
            num_label+=1
            return dict[item]
    newDF['天气状况白天'] = newDF['天气状况白天'].map(transform)
    newDF['天气状况晚上'] = newDF['天气状况晚上'].map(transform)

    train=train.drop(['天气状况'],axis=1)
    train=pd.concat([train,newDF],axis=1)
    return train

def air_temperature(train):
    newDF = train['气温'].str.split('/', expand=True)  # 测试包含哪些风力
    newDF.columns = ['气温白天', '气温晚上']
    global num_label
    num_label = 0
    dict = {}
    def transform(item):
        global num_label
        if item in dict:
            return dict[item]
        else:
            dict[item] = num_label
            num_label += 1
            return dict[item]

    newDF['气温白天'] = newDF['气温白天'].map(transform)
    newDF['气温晚上'] = newDF['气温晚上'].map(transform)

    train = train.drop(['气温'], axis=1)
    train = pd.concat([train, newDF], axis=1)
    return train

def wind(train):
    def transform(item):
        if '5-6' in item:
            return 3
        elif '4-5' in item:
            return 2
        elif '3-4' in item:
            return 1
        else:
            return 0
    #data=train[train['风力风向'].str.contains('6-7')] #测试包含哪些风力
    train['风力风向'] = train['风力风向'].map(transform)
    return train

def label(train):
    dict={'优': 0, '良': 1, '轻度污染': 2,'中度污染': 3, '重度污染': 4,  '严重污染': 5}
    def transform(item):
        return dict[item]
    #去除质量等级中是'无'的标签
    train = train[train['质量等级']!='无']
    train['质量等级']=train['质量等级'].map(transform)

    train = train.drop(['日期', '天气状况白天', '天气状况晚上'], axis=1)
    train = train.drop([ '气温白天','气温晚上'], axis=1)
    train = train.drop(['风力风向'], axis=1)

    y=train['质量等级']
    X=train.drop(['质量等级'],axis=1)
    train_x, test_x,train_y, test_y= train_test_split(X, y, test_size=.33, random_state=0)
    ## transforming "train_x"
    train_x = StandardScaler().fit_transform(train_x)
    ## transforming "test_x"
    test_x = StandardScaler().fit_transform(test_x)
    data=(train_x, train_y, test_x, test_y)

    return data,dict

def modelAll(model,name,*data):
    (x_train, y_train, x_test, y_test)=data
    model.fit(x_train, y_train)
    print(name)
    print('train accuracy is:{} '.format(model.score(x_train, y_train)))
    y_pred=model.predict(x_test)
    print('test accuracy is:{}'.format(accuracy_score(y_test, y_pred)))
    print('test precision_score is:{}'.format(precision_score(y_test, y_pred,average='macro')))
    print('test recall_score is:{}'.format(recall_score(y_test, y_pred,average='macro')))
    print('test f1_score is:{}'.format(f1_score(y_test, y_pred,average='macro')))

def predict():
    data=data_loader()
    data=weather_condition(data)
    data=air_temperature(data)
    data=wind(data)
    data,dict=label(data)
    print('类别标签如下')
    print(dict)

    model=LogisticRegression()
    modelAll(model,'LogisticR',*data)

    model = SVC(cache_size=200, class_weight=None, coef0=0.0, C=1,
              decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
    modelAll(model, 'SVM', *data)

    model = GradientBoostingClassifier()
    modelAll(model, 'GradientBoosting', *data)

if __name__=='__main__':
    predict()