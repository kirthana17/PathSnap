from pathlib import Path
import unittest
from src.pathsnap.io import load_trip
from src.pathsnap.features import extract_candidates,FEATURES
from src.pathsnap.decoder import viterbi_decode

ROOT=Path("data/raw")

class PathSnapSmokeTests(unittest.TestCase):
    def test_trip_loading(self):
        trip=load_trip(ROOT/"student_test"/"00000007")
        self.assertGreater(len(trip["track"]),0)
        self.assertEqual(list(trip["track"].columns),["lon","lat","time"])
    def test_candidate_features(self):
        trip=load_trip(ROOT/"student_test"/"00000007")
        df,_,track=extract_candidates(trip,k=8,labelled=False,observation_step=10)
        self.assertEqual(len(df),len(track)*8)
        self.assertTrue(set(FEATURES).issubset(df.columns))
    def test_viterbi_length(self):
        trip=load_trip(ROOT/"student_test"/"00000007")
        df,_,track=extract_candidates(trip,k=8,labelled=False,observation_step=10)
        df["probability"]=0.5
        pred=viterbi_decode(df,trip["arcs"].to_numpy(int))
        self.assertEqual(len(pred),len(track))

if __name__=="__main__":unittest.main()
