from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class Model:
    def __init__(self, model_name: str, pipeline: Pipeline):
        self.model_name = model_name
        self.scaler = StandardScaler()
        self.model = pipeline
        
    def fit(self, X, y):
        self.scaler.fit(y)
        y_true = self.scaler.transform(y).ravel()
        self.model.fit(X, y_true)

    def predict(self, X):
        y_pred = self.model.predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        return self.scaler.inverse_transform(y_pred)

    def get_score(self, X, y):
        y_true = self.scaler.transform(y).ravel()
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, r2