import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile)

    def run(self):
        self.processed_data = self.raw_input

        # split the data into input/output and train/test
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        '''
        Encoding the output class label
        le = LabelEncoder()
        # Using .fit_transform function to fit label
        # encoder and return encoded label
        label = le.fit_transform(self.processed_data['Classes'])
        self.processed_data.drop("Classes", axis=1, inplace=True)
        # Appending the array to our dataFrame
        #self.processed_datawith column name 'Purchased'
        self.processed_data["Classes"] = label
        '''
        Y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

        # standardize the data relative to their attributies
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        self.buildModel(X_train, X_test, Y_train, Y_test)

        return 0

    def buildModel(self, X_train, X_test, Y_train, Y_test):
        # Below are the hyperparameters used for the model evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and keep track of the performancs metrics
        count = 0
        # build all possible models from all the hyperparameters
        for act in activations:
          for lr in learning_rate:
            for nhl in num_hidden_layers:
              for mi in max_iterations:
                # train and test the model
                hiddenLayers = (13,) * nhl
                mlp = MLPClassifier(learning_rate_init = lr, max_iter = mi, hidden_layer_sizes=hiddenLayers, activation = act, random_state=1)
                mlp.fit(X_train,Y_train)
                predict_train = mlp.predict(X_train)
                predict_test = mlp.predict(X_test)
                
                # measure the model and display results for training data
                print("Activation:", act, "Learning Rate:", lr, "Max Iterations:", mi, "Hidden Layers:", nhl)
                print("Training Data:")
                print("Confusion Matrix:")
                print(confusion_matrix(Y_train,predict_train))
                print("Classification Report:")
                print(classification_report(Y_train,predict_train))
                
                # measure the model and display results for testing data
                print("Testing Data:")
                print("Confusion Matrix:")
                print(confusion_matrix(Y_test,predict_test))
                print("Classification Report:")
                print(classification_report(Y_test,predict_test))
                
                print("------------------------------------------------------")
                count += 1

        print(count, "Combinations")

        return 0

if __name__ == "__main__":
    neural_network = NeuralNet("https://raw.githubusercontent.com/ismailahmed0/Machine-Learning/main/Neural%20Network/Algerian_forest_fires_dataset_UPDATE.csv") # put in path to your file
    neural_network.run()