"""Candidate-road feature extraction.

Arc IDs are the zero-based row positions in .arcs, as specified by the dataset.
Reference .route values are used only to create training labels / evaluate the
student-test partition. They are never passed to the prediction function.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_R=6371000.0
FEATURES=["distance_m","heading_difference","road_length_m","gps_speed_mps",
          "bearing_change","candidate_bearing_alignment","time_gap_s"]

def haversine_xy(lon,lat,lon0,lat0):
    x=np.radians(lon-lon0)*np.cos(np.radians((lat+lat0)/2))*EARTH_R
    y=np.radians(lat-lat0)*EARTH_R
    return x,y

def project(lon,lat,lon0,lat0):
    return haversine_xy(np.asarray(lon),np.asarray(lat),lon0,lat0)

def bearing(x1,y1,x2,y2):
    return (np.degrees(np.arctan2(x2-x1,y2-y1))+360)%360

def angle_diff(a,b):
    return np.abs((a-b+180)%360-180)

def point_segment_distance(px,py,ax,ay,bx,by):
    vx,vy=bx-ax,by-ay
    den=vx*vx+vy*vy
    if den==0: return np.hypot(px-ax,py-ay)
    t=np.clip(((px-ax)*vx+(py-ay)*vy)/den,0,1)
    return np.hypot(px-(ax+t*vx),py-(ay+t*vy))

def prepare_geometry(trip):
    nodes=trip["nodes"].to_numpy(float)
    lon0=float(trip["track"].lon.median());lat0=float(trip["track"].lat.median())
    nx,ny=project(nodes[:,0],nodes[:,1],lon0,lat0)
    arcs=trip["arcs"].to_numpy(int)
    valid=(arcs[:,0]>=0)&(arcs[:,0]<len(nodes))&(arcs[:,1]>=0)&(arcs[:,1]<len(nodes))
    arcs=arcs[valid]
    ax,ay=nx[arcs[:,0]],ny[arcs[:,0]]
    bx,by=nx[arcs[:,1]],ny[arcs[:,1]]
    mx,my=(ax+bx)/2,(ay+by)/2
    return {"lon0":lon0,"lat0":lat0,"arcs":arcs,"arc_ids":np.flatnonzero(valid),"ax":ax,"ay":ay,"bx":bx,"by":by,
            "length":np.hypot(bx-ax,by-ay),"bearing":bearing(ax,ay,bx,by),
            "tree":cKDTree(np.c_[mx,my]),"midpoints":np.c_[mx,my]}

def extract_candidates(trip, k=12, labelled=False, observation_step=1):
    track=trip["track"].copy().dropna().reset_index(drop=True)
    if observation_step>1: track=track.iloc[::observation_step].reset_index(drop=True)
    g=prepare_geometry(trip)
    px,py=project(track.lon,track.lat,g["lon0"],g["lat0"])
    kk=min(k,len(g["arcs"]))
    _,inds=g["tree"].query(np.c_[px,py],k=kk)
    if kk==1: inds=inds[:,None]
    times=track.time.to_numpy(float)
    gps_bear=np.zeros(len(track))
    if len(track)>1:
        gps_bear[1:]=bearing(px[:-1],py[:-1],px[1:],py[1:])
        gps_bear[0]=gps_bear[1]
    speed=np.zeros(len(track))
    if len(track)>1:
        dt=np.maximum(np.diff(times),1e-3)
        speed[1:]=np.hypot(np.diff(px),np.diff(py))/dt
        speed[0]=speed[1]
    bearing_change=np.zeros(len(track))
    if len(track)>1: bearing_change[1:]=angle_diff(gps_bear[1:],gps_bear[:-1])
    # Broadcast candidate geometries for all observations at once.
    ax,ay=g["ax"][inds],g["ay"][inds]
    bx,by=g["bx"][inds],g["by"][inds]
    vx,vy=bx-ax,by-ay
    den=np.maximum(vx*vx+vy*vy,1e-12)
    wx=px[:,None]-ax; wy=py[:,None]-ay
    frac=np.clip((wx*vx+wy*vy)/den,0,1)
    dist=np.hypot(wx-frac*vx,wy-frac*vy)
    cand_bear=g["bearing"][inds]
    hd=angle_diff(gps_bear[:,None],cand_bear)
    ref_arc=np.full(len(track),-1,dtype=int)
    if labelled and trip["route"] is not None:
        raw=trip["route"].arc_id.to_numpy(int)
        valid_orig=g["arc_ids"]
        ref_ids=np.unique(raw[np.isin(raw,valid_orig)])
        if len(ref_ids):
            ref_geom=np.searchsorted(valid_orig,ref_ids)
            axr,ayr=g["ax"][ref_geom],g["ay"][ref_geom]
            bxr,byr=g["bx"][ref_geom],g["by"][ref_geom]
            vxr,vyr=bxr-axr,byr-ayr
            denr=np.maximum(vxr*vxr+vyr*vyr,1e-12)
            wxr=px[:,None]-axr[None,:];wyr=py[:,None]-ayr[None,:]
            fracr=np.clip((wxr*vxr[None,:]+wyr*vyr[None,:])/denr[None,:],0,1)
            dref=np.hypot(wxr-fracr*vxr[None,:],wyr-fracr*vyr[None,:])
            ref_arc=ref_ids[np.argmin(dref,axis=1)]
    n=len(track)
    frame=pd.DataFrame({
        "observation":np.repeat(np.arange(n),kk),
        "arc_id":g["arc_ids"][inds].reshape(-1),
        "distance_m":dist.reshape(-1),
        "heading_difference":hd.reshape(-1),
        "road_length_m":g["length"][inds].reshape(-1),
        "gps_speed_mps":np.repeat(speed,kk),
        "bearing_change":np.repeat(bearing_change,kk),
        "candidate_bearing_alignment":np.cos(np.radians(hd)).reshape(-1),
        "time_gap_s":np.repeat(np.r_[1.0,np.maximum(np.diff(times),0)],kk),
        "label":(g["arc_ids"][inds]==ref_arc[:,None]).astype(int).reshape(-1),
        "reference_arc":np.repeat(ref_arc,kk),
        "lon":np.repeat(track.lon.to_numpy(float),kk),
        "lat":np.repeat(track.lat.to_numpy(float),kk),
        "time":np.repeat(times,kk)
    })
    if not labelled:
        frame["label"]=-1
        frame["reference_arc"]=-1
    return frame,g,track
