from training import *
from utils_xgb import *
from collections import Counter
import time


if __name__=="__main__":

    max_depth = 3
    n_estimators = 200
    learning_rate = 0.3
    colsample_bytree = 0.79
    gamma = 0.0
    min_child_weight = 1
    subsample = 0.95
    test_size = 0.25

    x_tr, y_tr, x_ts, y_ts,data,scaler = readData("diabetes_cleaned_data.csv",test_size=test_size)

    model = trainModel(x_tr, y_tr, max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate,
                       colsample_bytree=colsample_bytree, gamma=gamma, min_child_weight=min_child_weight,
                       subsample=subsample)

    print(f"learning rate: {learning_rate}")
    print("Evaluation on the training set")
    train_acc_RF,train_cm_RF = evaluateModel(model, x_tr, y_tr)

    print("Evaluation on the test set")
    test_acc_RF,test_cm_RF = evaluateModel(model, x_ts, y_ts)

    time1 = time.time()

    proxMat_train, proxMat_w_train, weights = proximityMatrix(model,x_tr,learning_rate)
    proxMat_test, proxMat_w_test, weights_test = proximityMatrix(model,x_ts,learning_rate)


    visualizationRF = VisualizationRF(model,data,max_depth,n_estimators) # inizializing the class of the visualization toolkit

    df_xgb,df_depth,df_perc,df_nodes,df_nodes_perc,weight_times = visualizationRF.feature_usage_level(weights)

    visualizationRF.heatmap_RF_featuredepth(df_perc, 'figure', show=True)

    visualizationRF.heatmap_nodes(df_nodes_perc, 'figure/nodes',df_xgb)
    visualizationRF.tree_heatmap("tree_heatmap",df_nodes_perc,'figure')



    proxMat_w_train.to_csv('proxMat_weighted_train.csv')
    proxMat_w_test.to_csv('proxMat_weighted_test.csv')
    proxMat_train.to_csv('proxMat_noWeighted_train.csv')
    proxMat_test.to_csv('proxMat_noWeighted_test.csv')
    weights.to_csv('weights_forFreq.csv')
    weight_times.to_csv('weights_times.csv')


    # To save all the info that I need for the surrogate model of RF
    training_data = pd.DataFrame(x_tr, columns=list(data.columns)[:-1])
    training_data['y'] = y_tr
    test_data = pd.DataFrame(x_ts, columns=list(data.columns)[:-1])
    test_data['y'] = y_ts
    training_data.to_csv('training_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)

    df_nodes.to_csv('df_nodes.csv', index=False)
    print('depth', df_depth)
    print('nodes', df_nodes)
    # np.savetxt('df_nodes.txt', df_nodes.values.T, fmt='%d')

    y_tr_pred = model.predict(x_tr)
    y_ts_pred = model.predict(x_ts)
    y_tr_pred = pd.DataFrame(y_tr_pred, columns=['y_pred'])
    y_ts_pred = pd.DataFrame(y_ts_pred, columns=['y_pred'])
    y_tr_pred.to_csv('y_train_pred.csv')
    y_ts_pred.to_csv('y_test_pred.csv')

    res_RF = {'train_acc_RF': [train_acc_RF], 'train_cm_RF': [train_cm_RF], 'test_acc_RF': [test_acc_RF],
              'test_cm_RF': [test_cm_RF]}
    df_res_RF = pd.DataFrame.from_dict(res_RF)

    df_res_RF.to_csv('./stat_RF.csv', index=False)

    y_tr_pred = model.predict_proba(x_tr)
    y_tr_pred = y_tr_pred.max(axis=1)
    y_ts_pred = model.predict_proba(x_ts)
    y_ts_pred = y_ts_pred.max(axis=1)

    y_tr_pred = pd.DataFrame(y_tr_pred, columns=['y_pred_proba'])
    y_ts_pred = pd.DataFrame(y_ts_pred, columns=['y_pred_proba'])
    y_tr_pred.to_csv('y_train_pred_proba.csv')
    y_ts_pred.to_csv('y_test_pred_proba.csv')

    time2 = time.time()

    print(f"Time took to ran the code: {time2-time1} seconds")