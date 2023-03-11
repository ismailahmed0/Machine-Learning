import numpy as np
import pandas as pd  
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor

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

  # Train the Linear Regression model
  print("Training Data Results:")
  model = SGDRegressor(alpha=0.0000001, eta0=.1, max_iter = 100, tol = 0.000001)
  model.fit(X_train, Y_train)
  Y_train_predict = model.predict(X_train)
  modelMeasure(Y_train, Y_train_predict)
  print("Bias: ", model.intercept_, "\nCoefficients: ", model.coef_)

  print("--------------------------------------")

  # Test the Linear Regression model
  print("Testing Data Results:")
  Y_test_predict = model.predict(X_test)
  modelMeasure(Y_test, Y_test_predict)
  print("Bias: ", model.intercept_, "\nCoefficients: ", model.coef_)

main()
