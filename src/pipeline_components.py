import pandas as pd 
import numpy as np 
import os 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, r2_score
import joblib 
import kfp 
from kfp.dsl import component, Output, Input, Dataset, Model, Metrics, pipeline
from kfp import compiler

@component(base_image="python:3.10") 
def data_extraction(dvc_repo_url:str, dvc_path:str, output_data:Output[Dataset]): 
    import subprocess
    cmd = ["dvc", "get", dvc_repo_url, dvc_path, "-o", output_data.path] 
    subprocess.run(cmd,check=True) 

@component(base_image="python:3.10")
def data_preprocessing(raw_data:Input[Dataset], XtrainOut: Output[Dataset], XtestOut: Output[Dataset], YtrainOut:Output[Dataset], YtestOut:Output[Dataset]): 
    df = pd.read_csv(raw_data.path) 
    x = df.drop("MEDV",axis=1) 
    y = df['MEDV']
    x = StandardScaler().fit_transform(x) 
    Xtrain,Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.2, random_state=42) 
    pd.DataFrame(Xtrain).to_csv(XtrainOut.path,index=False) 
    pd.DataFrame(Xtest).to_csv(XtestOut.path,index=False) 
    pd.DataFrame(Ytrain).to_csv(YtrainOut.path,index=False) 
    pd.DataFrame(Ytest).to_csv(YtestOut.path,index=False) 

@component(base_image="python:3.10")
def train_model(Xtrain:Input[Dataset], Ytrain:Input[Dataset], modelOut:Output[Model]): 
    xtrain = pd.read_csv(Xtrain.path) 
    ytrain = pd.read_csv(Ytrain.path) 
    model = RandomForestRegressor(n_estimators=200,random_state=42) 
    model.fit(xtrain, np.ravel(ytrain))
    joblib.dump(model, modelOut.path)

@component(base_image="python:3.10")
def model_evaluation(modelin:Input[Model], Xtest:Input[Dataset], Ytest:Input[Dataset], metricsOut:Output[Metrics]): 
    model = joblib.load(modelin.path)
    xtest = pd.read_csv(Xtest.path) 
    ytest = pd.read_csv(Ytest.path) 
    preds = model.predict(xtest) 
    mse = mean_squared_error(ytest, preds)
    r2 = r2_score(ytest, preds) 
    metricsOut.log_metric("mse", float(mse)) 
    metricsOut.log_metric("r2", float(r2))  

@pipeline(name="boston-housing")
def func(dvc_repo_url: str, dvc_path: str):
    data = data_extraction(dvc_repo_url=dvc_repo_url, dvc_path=dvc_path)
    preprocess_task = data_preprocessing(raw_data=data.outputs["output_data"])
    train_task = train_model(Xtrain=preprocess_task.outputs["XtrainOut"],Ytrain=preprocess_task.outputs["YtrainOut"])
    model_evaluation(modelin=train_task.outputs["modelOut"],Xtest=preprocess_task.outputs["XtestOut"],Ytest=preprocess_task.outputs["YtestOut"])


if __name__ == "__main__": 
    compiler.Compiler().compile(
        pipeline_func=func, 
        package_path="components/boston_housing_pipeline.yaml"
    )