from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .io import iter_trips

def explore(data_root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for part in ("train","student_test"):
        for t in iter_trips(data_root, part):
            x,y,z=t["track"]["lon"],t["track"]["lat"],t["track"]["time"]
            dt=z.diff()
            arcs=t["arcs"].to_numpy(int); n_nodes=len(t["nodes"])
            valid=(arcs[:,0]>=0)&(arcs[:,0]<n_nodes)&(arcs[:,1]>=0)&(arcs[:,1]<n_nodes)
            valid_arcs=arcs[valid]
            graph=coo_matrix((np.ones(len(valid_arcs)),(valid_arcs[:,0],valid_arcs[:,1])),shape=(n_nodes,n_nodes)).tocsr()
            ncomp,labels=connected_components(graph,directed=False,return_labels=True)
            degree=np.asarray((graph+graph.T).sum(axis=1)).ravel()
            rows.append({
                "partition":part,"trip_id":t["trip_id"],
                "gps_observations":len(t["track"]),
                "nodes":len(t["nodes"]),"arcs":len(t["arcs"]),
                "invalid_arc_rows":int((~valid).sum()),
                "isolated_nodes":int((degree==0).sum()),
                "weakly_connected_components":int(ncomp),
                "route_arcs":len(t["route"]) if t["route"] is not None else 0,
                "missing_gps_cells":int(t["track"].isna().sum().sum()),
                "invalid_coordinate_rows":int((~x.between(-180,180)|~y.between(-90,90)).sum()),
                "non_increasing_timestamps":int((dt.dropna()<=0).sum()),
                "median_interval_s":float(dt.dropna().median()) if dt.notna().any() else None,
                "large_gaps_gt5s":int((dt>5).sum()),
                "lon_min":float(x.min()),"lon_max":float(x.max()),
                "lat_min":float(y.min()),"lat_max":float(y.max()),
                "track_bytes":(data_root/part/t["trip_id"]/f'{t["trip_id"]}.track').stat().st_size
            })
    df=pd.DataFrame(rows)
    df.to_csv(out/"dataset_inventory.csv",index=False)
    summary=df.groupby("partition").agg(
        trips=("trip_id","count"),
        gps_observations_median=("gps_observations","median"),
        gps_observations_min=("gps_observations","min"),
        gps_observations_max=("gps_observations","max"),
        nodes_median=("nodes","median"),arcs_median=("arcs","median"),
        route_arcs_median=("route_arcs","median"),
        missing_gps_cells=("missing_gps_cells","sum"),
        large_gaps_gt5s=("large_gaps_gt5s","sum")
    ).reset_index()
    summary.to_csv(out/"dataset_summary.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,5))
    for part,grp in df.groupby("partition"):
        ax.hist(grp["gps_observations"],bins=12,alpha=.6,label=part)
    ax.set(title="GPS observations per trip",xlabel="Observations",ylabel="Trip count")
    ax.legend();fig.tight_layout();fig.savefig(out/"gps_observations_distribution.png",dpi=160);plt.close(fig)
    # Representative trajectory plots from both permitted partitions.
    for part in ("train","student_test"):
        sample=df[df.partition==part].iloc[0]
        trip=next(t for t in iter_trips(data_root,part) if t["trip_id"]==sample.trip_id)
        fig,ax=plt.subplots(figsize=(7,6))
        ax.plot(trip["track"].lon,trip["track"].lat,"b-",linewidth=1,label="GPS trajectory")
        ax.scatter(trip["track"].lon.iloc[::max(1,len(trip["track"])//80)],
                   trip["track"].lat.iloc[::max(1,len(trip["track"])//80)],s=8,label="GPS samples")
        ax.set(xlabel="Longitude",ylabel="Latitude",title=f"{part} trip {sample.trip_id}")
        ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(out/f"{part}_{sample.trip_id}_trajectory.png",dpi=160);plt.close(fig)
    return df,summary

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/raw"))
    p.add_argument("--out",type=Path,default=Path("results/eda"))
    a=p.parse_args(); d,s=explore(a.data,a.out)
    print(s.to_string(index=False))
