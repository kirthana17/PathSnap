from pathlib import Path
import argparse
from .maps import create_map
p=argparse.ArgumentParser()
p.add_argument("--data",type=Path,default=Path("data/raw"))
p.add_argument("--partition",default="student_test")
p.add_argument("--trip-id",required=True)
p.add_argument("--out",type=Path,default=Path("maps"))
a=p.parse_args()
out=a.out/f"{a.partition}_{a.trip_id}.html"
create_map(a.data,a.partition,a.trip_id,out)
print(out)
