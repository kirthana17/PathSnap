from pathlib import Path
import optuna,joblib,json,time
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from .train import build_data
from .features import FEATURES

def main(data=Path("data/raw"),out=Path("results"),artifacts=Path("artifacts"),trials=20):
    out.mkdir(parents=True,exist_ok=True);artifacts.mkdir(parents=True,exist_ok=True)
    df=build_data(data,"train",step=30)
    chunks=[]
    for tid,g in df.groupby("trip_id"):
        pos=g[g.label==1];neg=g[g.label==0]
        chunks.append(pd.concat([pos.sample(n=min(50,len(pos)),random_state=42),
          neg.sample(n=min(150,len(neg)),random_state=42)]))
    df=pd.concat(chunks,ignore_index=True)
    X=df[FEATURES];y=df.label.astype(int);groups=df.trip_id
    cv=GroupKFold(n_splits=3)
    def objective(trial):
        params={"n_estimators":trial.suggest_int("n_estimators",50,120),
          "max_depth":trial.suggest_int("max_depth",8,20),
          "min_samples_leaf":trial.suggest_int("min_samples_leaf",1,8),
          "max_features":trial.suggest_categorical("max_features",["sqrt",0.7,1.0]),
          "class_weight":"balanced_subsample","n_jobs":-1,"random_state":42}
        model=make_pipeline(SimpleImputer(),RandomForestClassifier(**params))
        score=cross_val_score(model,X,y,groups=groups,cv=cv,scoring="f1_macro",n_jobs=1).mean()
        return float(score)
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective,n_trials=trials)
    study.trials_dataframe().to_csv(out/"bayesian_trials.csv",index=False)
    best={"n_estimators":study.best_params["n_estimators"],"max_depth":study.best_params["max_depth"],
      "min_samples_leaf":study.best_params["min_samples_leaf"],"max_features":study.best_params["max_features"],
      "class_weight":"balanced_subsample","n_jobs":-1,"random_state":42}
    model=make_pipeline(SimpleImputer(),RandomForestClassifier(**best)).fit(X,y)
    joblib.dump({"model":model,"features":FEATURES,"model_name":"Optuna-tuned Random Forest"},artifacts/"tuned_model.joblib")
    with open(out/"tuning_summary.json","w") as f:json.dump({"trials":trials,"best_cv_macro_f1":study.best_value,"best_params":best},f,indent=2)
    print("Best CV Macro F1:",study.best_value);print(best)
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,default=Path("data/raw"));p.add_argument("--out",type=Path,default=Path("results"));p.add_argument("--artifacts",type=Path,default=Path("artifacts"));p.add_argument("--trials",type=int,default=20)
    a=p.parse_args();main(a.data,a.out,a.artifacts,a.trials)
