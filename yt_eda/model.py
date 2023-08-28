from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class Model:
    """
    Model class to handle training, prediction, and evaluation tasks.
    
    Attributes:
    - data (Dataframe): Pandas Dataframe on which the model is going to be trained
    - target (str): Target variable name.
    
    Methods:
    - train: Fits the model on the training data.
    - predict: Makes predictions using the trained model.
    - evaluate: Evaluates the model's performance on a test dataset.
    """

    def __init__(self, data, target):
        """
        Initialize the Model object with an ML model and target variable.
        
        Parameters:
        - model (object): Machine learning model to be used.
        - target (str): Name of the target variable column.
        """
        self.data = data
        self.target = target
        self.model = RandomForestRegressor(n_estimators=100)
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()

    def _prepare_data(self):
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        """
        Fit the model on the training data.
        """
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X=None):
        """
        Predict using the trained model on new data.
        
        Parameters:
        - X (pd.DataFrame): New data features.
        
        Returns:
        - np.array: Predicted values.
        """
        if X is None:
            X = self.X_test
        return self.model.predict(X)

    def evaluate(self):
        """
        Evaluate the model's performance using the test data.
        
        This method computes the Mean Squared Error (MSE) and R^2 score for the model's 
        predictions on the test data. It assumes that the model has already been trained 
        and that the test data is available as an instance variable.
        
        Returns:
        - tuple: A tuple containing the Mean Squared Error (MSE) and R^2 score.
        
        Example:
        --------
        mse, r2 = model_instance.evaluate()
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")
        """
        predictions = self.predict()
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        return mse, r2

# Usage:
# model_instance = Model(engineered_data, 'avg_yearly_earnings')
# model_instance.train()
# mse, r2 = model_instance.evaluate()
# print(f"Mean Squared Error: {mse}")
# print(f"R^2 Score: {r2}")
