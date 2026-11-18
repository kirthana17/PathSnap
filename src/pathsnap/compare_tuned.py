from pathlib import Path
import time,joblib,json
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,f1_score,log_loss
from .train import build_data
from .features import FEATURES
from .io import load_trip

def continuity(df,data):
    vals=[]
    for tid,g in df.groupby("trip_id"):
        arcs=load_trip(data/"student_test"/tid)["arcs"].to_numpy(int)
        ids=g.sort_values("observation").arc_id.astype(int).to_numpy()
        vals.extend([int(arcs[a,1]==arcs[b,0]) for a,b in zip(ids[:-1],ids[1:]) if 0<=a<len(arcs) and 0<=b<len(arcs)])
    return float(np.mean(vals)) if vals else 0.

def evaluate(model,df,data,name):
    X=df[FEATURES]; y=df.label.astype(int)
    start=time.perf_counter();probs=model.predict_proba(X)[:,1];elapsed=time.perf_counter()-start
    d=df[["trip_id","observation","arc_id","reference_arc"]].copy();d["probability"]=probs
    chosen=d.loc[d.groupby(["trip_id","observation"]).probability.idxmax()].sort_values(["trip_id","observation"])
    return {"model":name,"student_top1_accuracy":accuracy_score(chosen.reference_arc,chosen.arc_id),
      "student_macro_f1":f1_score(chosen.reference_arc,chosen.arc_id,average="macro",zero_division=0),
      "route_continuity":continuity(chosen,data),"candidate_log_loss":log_loss(y,probs,labels=[0,1]),
      "inference_seconds":elapsed,"observations":len(chosen)}

def main(data=Path("data/raw"),out=Path("results"),artifacts=Path("artifacts")):
    train=build_data(data,"train",step=30);test=build_data(data,"student_test",step=5)
    parts=[]
    for tid,g in train.groupby("trip_id"):
        pos=g[g.label==1];neg=g[g.label==0]
        parts.append(pd.concat([pos.sample(n=min(50,len(pos)),random_state=42),
          neg.sample(n=min(150,len(neg)),random_state=42)]))
    train=pd.concat(parts,ignore_index=True)
    X=train[FEATURES];y=train.label.astype(int);groups=train.trip_id
    cv=GroupKFold(n_splits=3)
    baseline=make_pipeline(SimpleImputer(),RandomForestClassifier(n_estimators=120,max_depth=18,
      min_samples_leaf=3,class_weight="balanced_subsample",n_jobs=-1,random_state=42))
    baseline_cv=float(cross_val_score(baseline,X,y,groups=groups,cv=cv,scoring="f1_macro").mean())
    baseline.fit(X,y)
    tuned=joblib.load(artifacts/"tuned_model.joblib")["model"]
    tuned_cv=float(json.load(open(out/"tuning_summary.json"))["best_cv_macro_f1"])
    results=[]
    for m,n,cvscore in [(baseline,"Untuned Random Forest",baseline_cv),(tuned,"Optuna-tuned Random Forest",tuned_cv)]:
        r=evaluate(m,test,data,n);r["cv_macro_f1"]=cvscore;results.append(r)
    pd.DataFrame(results).to_csv(out/"tuned_vs_untuned.csv",index=False)
    print(pd.DataFrame(results).to_string(index=False))
if __name__=="__main__":
    main()
