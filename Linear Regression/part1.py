import numpy as np
import pandas as pd  
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# add a columns of 1s to the first column of the input training matrix to account for the bias/intercept
def addOneToInput(inputMatrix, m):
  oneMatrix = np.ones((m,1)) # 1202 rows and 1 column of 1s
  mergedMatrix = np.hstack((oneMatrix, inputMatrix)) # 1202x6 array
  return mergedMatrix

# gradient function for the vector
def gradientFunction(w, x, y, m):
  error = (np.dot(x, w) - y) / m
  gradient = np.dot(error, x)
  return gradient

# does gradient desscent for the model
def gradient_descent(gradient, X, Y, m, start, learn_rate = .1, n_iter=100, tolerance=1e-06): # tolerance is delta
  vector = start # start is a guess
  for _ in range(n_iter):
    diff = -learn_rate * gradient(vector, X, Y, m)
    if np.all(np.abs(diff) <= tolerance):
      break
    vector += diff
  return vector

# display model accuracy
def modelMeasure(Y_input, Y_output_predict):
  rmse = (np.sqrt(mean_squared_error(Y_input, Y_output_predict)))
  r2 = r2_score(Y_input, Y_output_predict)
  print("RMSE:", rmse)
  print("R^2:", r2)

def main():
  # read the dataset
  df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat", header = None, delimiter='\t')
  df.head()

  # get the locations
  X = df.iloc[:, :-1] 
  Y = df.iloc[:, 5]

  # normalize
  s = StandardScaler()
  X = pd.DataFrame(s.fit(X).fit_transform(X))

  # split the dataset into training/testing 80/20 ratio
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
  m1, n1 = X_train.shape # for generality, m = rows
  m2, n2 = X_test.shape

  # random theta weights for starting out
  thetas = [1, 2, 3, 4, 5, 6]

  # Train the Linear Regression model
  print("Training Data Results:")
  X_train_new = addOneToInput(X_train, m1)
  X_train_vector = gradient_descent(gradientFunction, X_train_new, Y_train, m1, start=thetas) # gets accurate weights as much as possible
  Y_train_predict = np.dot(X_train_new, X_train_vector)
  modelMeasure(Y_train, Y_train_predict)
  print("Bias and Coefficients: ", X_train_vector)

  print("--------------------------------------")

  # Test the Linear Regression model
  print("Testing Data Results:")
  X_test_new = addOneToInput(X_test, m2)
  X_test_vector = gradient_descent(gradientFunction, X_test_new, Y_test, m2, start=thetas)
  Y_test_predict = np.dot(X_test_new, X_test_vector)
  modelMeasure(Y_test, Y_test_predict)
  print("Bias and Coefficients: ", X_train_vector)

main()

