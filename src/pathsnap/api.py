from pathlib import Path
import joblib
from fastapi import FastAPI,HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from .io import load_trip
from .features import extract_candidates
from sklearn.metrics import accuracy_score,f1_score
from .decoder import viterbi_decode
from .maps import create_map
import numpy as np

ROOT=Path("data/raw");MODEL=Path("artifacts/best_model.joblib")
app=FastAPI(title="PathSnap Map Matching API",version="1.0")
artifact=None
@app.on_event("startup")
def load_model():
    global artifact
    if MODEL.exists(): artifact=joblib.load(MODEL)

class PredictRequest(BaseModel):
    trip_id:str
    partition:str="student_test"

@app.get("/health")
def health():
    return {"status":"ok","model_loaded":artifact is not None}

@app.get("/trips")
def trips():
    # Prefer the reserved evaluator partition when installed on the receiving laptop.
    partition="evaluator_test" if (ROOT/"evaluator_test").exists() else "student_test"
    p=ROOT/partition
    return {"partition":partition,"trip_ids":sorted(x.name for x in p.iterdir() if x.is_dir())}

@app.post("/predict")
def predict(req:PredictRequest):
    if req.partition not in {"student_test","evaluator_test"}:
        raise HTTPException(400,"Only reserved evaluation partitions can be requested.")
    if artifact is None: raise HTTPException(503,"Model not trained. Run training first.")
    folder=ROOT/req.partition/req.trip_id
    if not folder.exists(): raise HTTPException(404,"Trip not found")
    trip=load_trip(folder)
    cand,_,track=extract_candidates(trip,k=8,labelled=False,observation_step=1)
    probs=artifact["model"].predict_proba(cand[artifact["features"]])[:,1]
    cand["probability"]=probs
    chosen=cand.loc[cand.groupby("observation").probability.idxmax()].sort_values("observation")
    route=viterbi_decode(cand,trip["arcs"].to_numpy(int))
    # Evaluation is computed server-side after inference. Reference labels never
    # enter the classifier inputs and are never returned to the browser.
    labelled,_,_=extract_candidates(trip,k=8,labelled=True,observation_step=1)
    refs=labelled.groupby("observation").reference_arc.first().to_numpy(int)
    n=min(len(refs),len(route))
    metrics={}
    if n:
        metrics={"top1_accuracy":float(accuracy_score(refs[:n],np.asarray(route[:n]))),
                 "macro_f1":float(f1_score(refs[:n],np.asarray(route[:n]),average="macro",zero_division=0))}
    ids=np.asarray(route,dtype=int);aa=trip["arcs"].to_numpy(int)
    connected=[int(aa[a,1]==aa[b,0]) for a,b in zip(ids[:-1],ids[1:]) if 0<=a<len(aa) and 0<=b<len(aa)]
    metrics["route_continuity"]=float(np.mean(connected)) if connected else 0.0
    return {"trip_id":req.trip_id,"observations":len(track),
      "predicted_arc_ids":route,"classifier_arc_ids":chosen.arc_id.astype(int).tolist(),
      "evaluation_metrics":metrics,"model":artifact["model_name"]}


@app.get("/map/{partition}/{trip_id}",response_class=HTMLResponse)
def trip_map(partition:str,trip_id:str):
    if partition not in {"student_test","evaluator_test"}:
        raise HTTPException(400,"Invalid partition")
    folder=ROOT/partition/trip_id
    if not folder.exists(): raise HTTPException(404,"Trip not found")
    # Never expose evaluator reference routes in the browser.
    out=Path("maps")/f"{partition}_{trip_id}_api.html"
    create_map(ROOT,partition,trip_id,out,MODEL,include_reference=False)
    return HTMLResponse(out.read_text(encoding="utf-8"))
