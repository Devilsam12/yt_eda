from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class Model:

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.model = RandomForestRegressor(n_estimators=100)
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()

    def _prepare_data(self):
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X=None):
        if X is None:
            X = self.X_test
        return self.model.predict(X)

    def evaluate(self):
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
