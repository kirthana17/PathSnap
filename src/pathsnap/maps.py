from pathlib import Path
import folium
from folium import FeatureGroup,LayerControl
from .io import load_trip
from .features import extract_candidates
import joblib
from .decoder import viterbi_decode

def create_map(data_root:Path,partition:str,trip_id:str,out:Path,model_path=Path("artifacts/best_model.joblib"),include_reference=True):
    trip=load_trip(data_root/partition/trip_id)
    t=trip["track"]; nodes=trip["nodes"].to_numpy(float); arcs=trip["arcs"].to_numpy(int)
    center=[float(t.lat.median()),float(t.lon.median())]
    m=folium.Map(location=center,zoom_start=14,control_scale=True)
    gps=FeatureGroup(name="Observed GPS",show=True)
    folium.PolyLine(list(zip(t.lat,t.lon)),color="blue",weight=3,opacity=.8).add_to(gps)
    for i,r in t.iloc[::max(1,len(t)//100)].iterrows():
        folium.CircleMarker([r.lat,r.lon],radius=2,color="blue",fill=True,
          popup=f"GPS observation {i}").add_to(gps)
    gps.add_to(m)
    road=FeatureGroup(name="Road network",show=False)
    # Restrict displayed geometry to a local bounding box around the trajectory.
    lo,hi=t.lon.min()-.01,t.lon.max()+.01; la,ha=t.lat.min()-.01,t.lat.max()+.01
    for u,v in arcs:
        if u<len(nodes) and v<len(nodes):
            a,b=nodes[u],nodes[v]
            if lo<=a[0]<=hi and la<=a[1]<=ha and lo<=b[0]<=hi and la<=b[1]<=ha:
                folium.PolyLine([[a[1],a[0]],[b[1],b[0]]],color="#999999",weight=1,opacity=.45).add_to(road)
    road.add_to(m)
    if include_reference and trip["route"] is not None:
        route=trip["route"].arc_id.to_numpy(int)
        ref=FeatureGroup(name="Reference route",show=True)
        pts=[]
        for aid in route:
            if 0<=aid<len(arcs):
                u,v=arcs[aid]
                if u<len(nodes) and v<len(nodes):
                    pts.extend([[nodes[u,1],nodes[u,0]],[nodes[v,1],nodes[v,0]]])
        if pts: folium.PolyLine(pts,color="green",weight=4,opacity=.8).add_to(ref)
        ref.add_to(m)
    if model_path.exists():
        artifact=joblib.load(model_path)
        candidates,_,_=extract_candidates(trip,k=8,labelled=False,observation_step=2)
        probs=artifact["model"].predict_proba(candidates[artifact["features"]])[:,1]
        candidates["probability"]=probs
        chosen=candidates.loc[candidates.groupby("observation").probability.idxmax()]
        pred=FeatureGroup(name="Predicted route (classifier)",show=True)
        p=[]
        for aid in chosen.arc_id.astype(int):
            if 0<=aid<len(arcs):
                u,v=arcs[aid]
                if u<len(nodes) and v<len(nodes):
                    p.extend([[nodes[u,1],nodes[u,0]],[nodes[v,1],nodes[v,0]]])
        if p: folium.PolyLine(p,color="red",weight=4,opacity=.85).add_to(pred)
        pred.add_to(m)
        # HMM/Viterbi route layer from the same candidate emissions.
        candidates["probability"]=probs
        decoded=viterbi_decode(candidates,arcs)
        hmm=FeatureGroup(name="Predicted route (HMM/Viterbi)",show=True)
        hp=[]
        for aid in decoded:
            if 0<=aid<len(arcs):
                u,v=arcs[aid]
                if u<len(nodes) and v<len(nodes):
                    hp.extend([[nodes[u,1],nodes[u,0]],[nodes[v,1],nodes[v,0]]])
        if hp: folium.PolyLine(hp,color="purple",weight=4,opacity=.85).add_to(hmm)
        hmm.add_to(m)
    LayerControl(collapsed=False).add_to(m)
    out.parent.mkdir(parents=True,exist_ok=True);m.save(str(out))
    return out
