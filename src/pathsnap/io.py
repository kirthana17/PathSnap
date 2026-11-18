from pathlib import Path
import pandas as pd
import numpy as np

def load_trip(folder: Path):
    tid = folder.name
    def read(ext):
        p = folder / f"{tid}.{ext}"
        return pd.read_csv(p, sep=r"\s+", header=None) if p.exists() else None
    track, nodes, arcs, route = (read(x) for x in ("track","nodes","arcs","route"))
    if track is None or nodes is None or arcs is None:
        raise ValueError(f"Trip {tid} is missing a required .track/.nodes/.arcs file")
    track.columns = ["lon","lat","time"]
    nodes.columns = ["lon","lat"]
    arcs.columns = ["from_node","to_node"]
    if route is not None:
        route.columns = ["arc_id"]
    return {"trip_id":tid,"track":track,"nodes":nodes,"arcs":arcs,"route":route}

def iter_trips(root: Path, partition: str):
    folder = root / partition
    for trip_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        yield load_trip(trip_dir)
