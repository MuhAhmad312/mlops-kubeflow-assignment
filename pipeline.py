from kfp import compiler 
from kfp.dsl import pipeline 
from src.pipeline_components import data_extraction, data_preprocessing, train_model, model_evaluation
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