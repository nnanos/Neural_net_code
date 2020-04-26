import numpy as np
import numpy.matlib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


def load_data_user_movie_matrix(file_name,x,z,y):

     centered_matrix = []

     nb_classes = 943
     targets = np.arange(943).reshape(-1)
     one_hot_encoded_users = np.eye(nb_classes)[targets]


     #creating the user-movie matrix
     pwd = os.getcwd()+'\\'+file_name
     data = pd.read_csv(pwd, sep='\t', names=['user', 'item', 'rating', 'timestamp'], header=None ,encoding='utf-8' )
     data = data.drop('timestamp', axis=1)

     #obtaning user-movie matrix from the movielens data
     matrix = data.pivot(index='user', columns='item', values='rating')

     #preprocessing...
     #fill in the missing columns of the corresponding matrix in order for the training-testing to work(because the output to our model is 1682)-------------------
     new_index_column = np.arange(1682) + 1
     matrix = matrix.reindex(new_index_column, axis=1)
     #--------------------------------------------------------------------------

     # obtainiing the x_test (because in each test fold the number of users are always less than 943 and we should know the indices)
     #the x_test contains the users that are going to be used to the testing phase
     if 'test' in file_name :
          tmp = matrix.index
          tmp = tmp.to_numpy()-1
          x_test = one_hot_encoded_users[tmp]
     #----------------------------------------------------------------------------


     mean_vector = np.nanmean(matrix, axis=1)

     if int(x) == 0 :
          #if we dont want our data to be centered
          matrix = matrix.to_numpy()
          centered_matrix = np.nan_to_num(matrix)

     else:
          #if we want our data to be centered...obtaining the centered_matrix
          matrix = matrix.to_numpy()
          mean_arr = np.transpose(np.matlib.repmat(np.transpose(mean_vector), matrix.shape[1], 1))
          centered_matrix = matrix - mean_arr
          # filling the empty entries with zeros
          centered_matrix = np.nan_to_num(centered_matrix)




     # filling the empty entries with the mean of every user's rating
     if int(y)==3:
          for i in range(centered_matrix.shape[0]):
               for j in range(centered_matrix.shape[1]):
                    if centered_matrix[i, j] == 0:
                         centered_matrix[i, j] = mean_vector[i]
                    else:
                         continue
     #----------------------------------------------------------------------

     elif int(y)==2 :
     # filling the empty entries with random values------------------------------
          for i in range(centered_matrix.shape[0]):
               for j in range(centered_matrix.shape[1]):
                    if centered_matrix[i,j] == 0:
                         centered_matrix[i,j] = np.random.randn()
                    else:
                         continue
     #----------------------------------------------------------------------------


     if int(z)==1:
          # scalling the data [min,max]->[0,1]
          scaler = MinMaxScaler()
          scaler.fit(centered_matrix)
          centered_matrix = scaler.transform(centered_matrix)
     else:
          centered_matrix = centered_matrix


     result_matrix=centered_matrix

     if 'test' in file_name:
          return result_matrix,x_test
     else:
         if int(x)==1:
             #return the mean arr in order to add this to our predicted data
            if int(z) == 1:
                #return scaler in order to inverse transform the predicted data
                return (result_matrix , scaler,mean_arr)
            else:
                return result_matrix,mean_arr
         else:
             if int(z) == 1:
                 # return scaler in order to inverse transform the predicted data
                 return (result_matrix, scaler)
             else:
                 return result_matrix
