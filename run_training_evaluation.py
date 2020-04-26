import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import load_movielens_data
from sklearn.model_selection import KFold
import sklearn
import numpy as np
import model_module
import keras
import data_processing_func
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot


x = input('enter 1 if you want your data to be centered and 0 if you dont')
l = input('enter 1 if you want your data to be scaled to the range [0,1](using sigmoid activation func) and 0 if you dont')
y = input(
    'enter 1 for filling the missing values with 0,\n 2 for filling the empty values with random values,\n 3 for filling the empty values with the mean ')


# one hot encoded users----
nb_classes = 943
targets = np.arange(943).reshape(-1)
one_hot_encoded_users = np.eye(nb_classes)[targets]
# ----------------------------------

#loading the user-movie rating matrix
if int(x)==1:
    if int(l)==1:
        data,data_scaler,mean_arr = load_movielens_data.load_data_user_movie_matrix('u.data',int(x),int(l),int(y))
    else:
        data,mean_arr = load_movielens_data.load_data_user_movie_matrix('u.data', int(x), int(l), int(y))
else:
    if int(l)==1:
        data,data_scaler = load_movielens_data.load_data_user_movie_matrix('u.data',int(x),int(l),int(y))
    else:
        data = load_movielens_data.load_data_user_movie_matrix('u.data', int(x), int(l), int(y))



#TRAINING and EVALUATING process WITH THE USER-MOVIE MATRIX SPLITTED ROW-WISE
#AND WITH SPLITTING THAT I DID CUSTOMELY (splitted the input and the output accordingly)-------------------------------------------------------

#keep a 10% holdout
x_main,x_holdout,y_main,y_holdout = sklearn.model_selection.train_test_split(one_hot_encoded_users,data,test_size = 0.10)

kf = KFold(n_splits=5)

KFold(n_splits=5, random_state=None, shuffle=False)
#model = model_module.create_model()
evaluation_for_each_fold = []
oos_y = []
oos_pred = []

tmp = 0
for train_index, test_index in kf.split(y_main):
    tmp+=1

    x_train, x_test = x_main[train_index], x_main[test_index]
    y_train, y_test = y_main[train_index], y_main[test_index]

    model = model_module.create_model(int(l))
    monitor = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=0, verbose=1, mode='auto')
    history = model.fit(x_train, y_train , validation_split=0.1, callbacks = [monitor] ,verbose=1 ,epochs=100)

    if tmp==1:
        mse_tmp_training = np.array([history.history['mean_squared_error']])
        mae_tmp_training = np.array([history.history['mean_absolute_error']])
    else:
        mse_tmp_training = np.concatenate((mse_tmp_training,np.array([history.history['mean_squared_error']])),axis=0)
        mae_tmp_training = np.concatenate((mae_tmp_training, np.array([history.history['mean_absolute_error']])), axis=0)


    pred_y_hat = model.predict(x_test)

    oos_y.append(y_test)
    oos_pred.append(pred_y_hat)

    #mse for each fold (after validation)
    score = np.sqrt(sklearn.metrics.mean_squared_error(pred_y_hat, y_test))

    evaluation_for_each_fold.append(score)


#printing the rmse for each fold
for i in range(5):
    print("fold"+str(i+1)+" score (RMSE): {} \n" .format( evaluation_for_each_fold[i] ))

#we concatenate all the validation y_tests from each fold(true values)
oos_y = np.concatenate(oos_y)
#we concatenate all the predictions from each fold (estimated values)
oos_pred = np.concatenate(oos_pred)
#MSE of the error matrix (predictions-y) totally
score = np.sqrt(sklearn.metrics.mean_squared_error(oos_pred, oos_y))
#holdout evaluation
final_pred = model.predict(x_holdout)
#final RMSE (holdout score)
final_score = np.sqrt(sklearn.metrics.mean_squared_error(final_pred, y_holdout))
print("cross validated score (RMSE): {} \n" .format( score ))
print("holdout score (RMSE): {}\n" .format(final_score))
print("holdout score (MAE): {}\n" .format(sklearn.metrics.mean_absolute_error(final_pred, y_holdout)))


pyplot.figure()
#plotting the mse/epoch (M.O.) for the training phase
pyplot.plot(np.mean(mse_tmp_training,axis=0))
pyplot.figure()
#plotting the mae/epoch (M.O.) for the training phase
pyplot.plot(np.mean(mae_tmp_training,axis=0))

#------------------------------------------------------------------------------------------


