from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

def readData(filename, test_size):
    data = pd.read_csv(filename)
    x = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]

    x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=test_size, stratify=y, random_state=10)

    scaler =  MinMaxScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_ts = scaler.transform(x_ts)

    return x_tr, y_tr, x_ts, y_ts, data, scaler

def trainModel(x, y, max_depth, n_estimators, learning_rate, colsample_bytree, gamma, min_child_weight, subsample):
    # Initialize the XGBoost classifier with the given parameters
    clf = XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        min_child_weight=min_child_weight,
        subsample=subsample,
        objective='binary:logistic',
        random_state=100
    )

    # Fit the model
    clf.fit(x, y)

    return clf

def trainModel_RF(x, y,
               max_depth,n_estimators,max_features=None):
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,class_weight='balanced',
                                  max_features=max_features,random_state=100)


    model.fit(x,y)

    return model


def evaluateModel(model, x, y):
    y_pred = model.predict(x)
    y_proba = model.predict_proba(x)[:, 1]  # Ensure your model has predict_proba and it's appropriate for AUC

    # Calculating metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')

    # Handling for AUC
    try:
        auc = roc_auc_score(y, y_proba)
    except ValueError:
        auc = "AUC not applicable"

    # Printing metrics
    print("Accuracy: {:.2f}".format(accuracy))
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("F1-Score: {:.2f}".format(f1))
    print("AUC: {}".format(auc))
    print(confusion_matrix(y, y_pred))

    return accuracy_score(y, y_pred),confusion_matrix(y, y_pred)