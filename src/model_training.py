import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
import json
from datetime import datetime


class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_data(self, data_path):
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Dataset shape: {df.shape}")
        return df
    
    def preprocess_data(self, df, test_size=0.2):
        print("Preprocessing data...")
        df = df.dropna()
        target_column = 'medv'
        if target_column not in df.columns:
            target_column = df.columns[-1]
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train, n_estimators=100, max_depth=10):
        print(f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        print(f"Training R² Score: {train_score:.4f}")
        
        return train_score
    
    def evaluate_model(self, X_test, y_test):
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        print(f"R² Score: {metrics['r2_score']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        
        return metrics
    
    def save_model(self, model_path='models/random_forest_model.pkl'):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def save_metrics(self, metrics, metrics_path='metrics/model_metrics.json'):
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        metrics['timestamp'] = datetime.now().isoformat()
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to {metrics_path}")


def main():
    DATA_PATH = 'data/raw_data.csv'
    MODEL_PATH = 'models/random_forest_model.pkl'
    METRICS_PATH = 'metrics/model_metrics.json'
    
    trainer = ModelTrainer(random_state=42)
    
    df = trainer.load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = trainer.preprocess_data(df, test_size=0.2)
    
    trainer.train_model(X_train, y_train, n_estimators=100, max_depth=10)
    
    metrics = trainer.evaluate_model(X_test, y_test)
    
    trainer.save_model(MODEL_PATH)
    trainer.save_metrics(metrics, METRICS_PATH)
    
    print("\n Training completed successfully!")


if __name__ == '__main__':
    main()
