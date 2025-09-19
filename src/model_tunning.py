import optuna
from optuna import trial
from hyperopt import hp
import mlflow
import joblib
import os, sys
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier

from logger import get_logger
from exception import MLException
from dotenv import load_dotenv, find_dotenv

from model_training import ModelTraining

load_dotenv(find_dotenv())

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.ERROR)

trainer = ModelTraining()

def champion_callback(study:optuna.study.Study, frozen_trial:optuna.trial.FrozenTrial):
    winner = study.user_attrs.get('winner',None)
    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            
            logger.info(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
              f"{improvement_percent: .4f}% improvement"
            )
        else:
            logger.info(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")
            
def objective():
    mlflow.sklearn.autolog()
    with mlflow.start_run(nested=True) as child_run:
        
        params = {
            "objective": "classification:roc_auc_score",
            "eval_metric": "roc_auc_score",
            "n_estimators": trial.suggest_int("n_estimators",10,100,log=True),  #("booster", ["gbtree", "gblinear", "dart"]),
            "learning_rate": trial.suggest_uniform("learning_rate", 0.1, 1.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 5, log=True),
        }
        
        roc_auc = trainer.run(params=params, hyperparameter_tunning=True)
        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", roc_auc)
        
    return roc_auc

def tune_parameters(experiment_name:str):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    run_name = "first_attempt"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True) as parent_run:
        study = optuna.create_study(direction='minimize')
        
        study.optimize(func=objective, n_trials=10, callbacks=[champion_callback])
        
        mlflow.log_params(study.best_params)
        mlflow.log_metric("roc_auc",study.best_value)
        
        mlflow.set_tags(
            tags={
                "project": "Cancer Prediction",
                "optimizer_engine": "optuna",
                "model_family": "GradientBoostClassifier",
                "feature_set_version": 1,
            }
        )

        _, model, signature = trainer.run(params=study.best_params)
        
        artifact_path = "model"

        X_train = joblib.load(os.path.join("artifacts/processed" , "X_train.pkl"))
        mlflow.sklearn.log_model(
            sk_model=model,
            name=artifact_path,
            signature=signature,
            model_format="ubj",
            metadata={"model_data_version": 1},
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)
    
if __name__=="__main__":
    tune_parameters(experiment_name="cancer_detection_experiment")