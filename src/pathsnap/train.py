
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from .io import iter_trips, load_trip
from .features import extract_candidates, FEATURES
from .decoder import viterbi_decode


def build_data(root, partition, step=3):
    chunks = []

    for trip in iter_trips(root, partition):
        df, _, _ = extract_candidates(
            trip,
            k=8,
            labelled=True,
            observation_step=step
        )

        df["trip_id"] = trip["trip_id"]
        chunks.append(df[df.label >= 0])

    return pd.concat(chunks, ignore_index=True)


def main(
    data=Path("data/raw"),
    out=Path("results"),
    artifacts=Path("artifacts")
):
    out.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    print("Building training candidate table...")

    train = build_data(data, "train", step=15)
    test = build_data(data, "student_test", step=5)

    # Cap each trip's candidate rows to keep cross-validation
    # computationally tractable.
    sampled = []

    for tid, grp in train.groupby("trip_id"):
        pos = grp[grp.label == 1]
        neg = grp[grp.label == 0]

        npos = min(len(pos), 125)
        nneg = min(len(neg), max(125, 4 * npos))

        sampled.append(
            pd.concat([
                pos.sample(n=npos, random_state=42) if npos else pos,
                neg.sample(n=nneg, random_state=42) if nneg else neg
            ])
        )

    train = pd.concat(sampled, ignore_index=True)

    train.to_csv(out / "train_candidates.csv", index=False)
    test.to_csv(out / "student_test_candidates.csv", index=False)

    X = train[FEATURES]
    y = train.label.astype(int)
    groups = train.trip_id

    Xt = test[FEATURES]
    yt = test.label.astype(int)

    # Five classification models.
    models = {
        "Logistic Regression": make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            LogisticRegression(
                max_iter=500,
                class_weight="balanced"
            )
        ),

        "KNN": make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=15,
                weights="distance"
            )
        ),

        "Decision Tree": make_pipeline(
            SimpleImputer(),
            DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42
            )
        ),

        "Random Forest": make_pipeline(
            SimpleImputer(),
            RandomForestClassifier(
                n_estimators=120,
                max_depth=18,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42
            )
        ),

        "Extra Trees": make_pipeline(
            SimpleImputer(),
            ExtraTreesClassifier(
                n_estimators=120,
                max_depth=18,
                min_samples_leaf=3,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )
        ),
    }

    cv = GroupKFold(n_splits=5)

    results = []
    fitted = {}

    for name, model in models.items():
        print("Training", name, flush=True)

        # Grouped cross-validation by trip ID.
        cv_start = time.perf_counter()

        scores = cross_val_score(
            model,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1
        )

        cv_seconds = time.perf_counter() - cv_start

        # Fit on the complete sampled training partition.
        fit_start = time.perf_counter()
        model.fit(X, y)
        training_seconds = time.perf_counter() - fit_start

        # Evaluate on student-test candidates.
        inference_start = time.perf_counter()
        probs = model.predict_proba(Xt)
        inference_seconds = time.perf_counter() - inference_start

        candidate_pred = model.predict(Xt)

        student_candidate_f1 = f1_score(
            yt,
            candidate_pred,
            average="macro",
            zero_division=0
        )

        # Observation-level prediction:
        # choose the candidate with the highest positive probability.
        eval_df = test[
            ["trip_id", "observation", "arc_id", "reference_arc"]
        ].copy()

        eval_df["probability"] = probs[:, 1]

        chosen = eval_df.loc[
            eval_df.groupby(
                ["trip_id", "observation"]
            ).probability.idxmax()
        ].sort_values(["trip_id", "observation"])

        acc = accuracy_score(
            chosen.reference_arc,
            chosen.arc_id
        )

        mf1 = f1_score(
            chosen.reference_arc,
            chosen.arc_id,
            average="macro",
            zero_division=0
        )

        # Route continuity.
        continuity = []

        for tid, gp in chosen.groupby("trip_id"):
            trip_obj = load_trip(data / "student_test" / tid)
            aa = trip_obj["arcs"].to_numpy(int)
            ids = gp.arc_id.astype(int).to_numpy()

            if len(ids) > 1:
                continuity.extend([
                    int(
                        0 <= a < len(aa)
                        and 0 <= b < len(aa)
                        and aa[a, 1] == aa[b, 0]
                    )
                    for a, b in zip(ids[:-1], ids[1:])
                ])

        route_cont = (
            float(np.mean(continuity))
            if continuity else 0.0
        )

        # Candidate miss rate.
        miss = []

        for (tid, obs), gg in test.groupby(
            ["trip_id", "observation"]
        ):
            miss.append(
                int(
                    gg.reference_arc.iloc[0]
                    not in set(gg.arc_id.astype(int))
                )
            )

        candidate_miss = (
            float(np.mean(miss))
            if miss else 1.0
        )

        results.append({
            "model": name,
            "cv_macro_f1_mean": float(scores.mean()),
            "cv_macro_f1_std": float(scores.std()),
            "student_top1_accuracy": float(acc),
            "student_macro_f1": float(mf1),
            "route_continuity": route_cont,
            "candidate_miss_rate": candidate_miss,
            "student_candidate_macro_f1": float(
                student_candidate_f1
            ),
            "student_log_loss": float(
                log_loss(yt, probs, labels=[0, 1])
            ),
            "cv_seconds": float(cv_seconds),
            "training_seconds": float(training_seconds),
            "inference_seconds": float(inference_seconds),
            "candidate_rows": len(train)
        })

        fitted[name] = model

    # Save model comparison.
    table = pd.DataFrame(results)
    table.to_csv(
        out / "model_comparison.csv",
        index=False
    )

    # Select using cross-validation Macro F1 only.
    # Student-test metrics are reserved for evaluation.
    best_name = table.sort_values(
        "cv_macro_f1_mean",
        ascending=False
    ).iloc[0]["model"]

    joblib.dump(
        {
            "model": fitted[best_name],
            "features": FEATURES,
            "model_name": best_name
        },
        artifacts / "best_model.joblib"
    )

    # Compare classifier-only predictions with HMM/Viterbi
    # for the selected model.
    best_model = fitted[best_name]
    best_probs = best_model.predict_proba(Xt)[:, 1]

    test_prob = test.copy()
    test_prob["probability"] = best_probs

    hmm_rows = []

    for tid, gp in test_prob.groupby("trip_id"):
        trip_obj = load_trip(data / "student_test" / tid)
        aa = trip_obj["arcs"].to_numpy(int)

        cls = gp.loc[
            gp.groupby("observation").probability.idxmax()
        ].sort_values("observation")

        decoded = viterbi_decode(gp, aa)

        refs = cls.reference_arc.astype(int).to_numpy()
        cp = cls.arc_id.astype(int).to_numpy()

        n = min(len(refs), len(decoded))

        for obs, (ref, cl, hm) in enumerate(
            zip(refs[:n], cp[:n], decoded[:n])
        ):
            hmm_rows.append({
                "trip_id": tid,
                "observation": obs,
                "reference_arc": ref,
                "classifier_arc": cl,
                "viterbi_arc": int(hm)
            })

    hmmdf = pd.DataFrame(hmm_rows)
    hmmdf.to_csv(
        out / "hmm_predictions.csv",
        index=False
    )

    if len(hmmdf):
        hcomp = []

        for col in ["classifier_arc", "viterbi_arc"]:
            continuity = []

            for tid, gp in hmmdf.groupby("trip_id"):
                aa = load_trip(
                    data / "student_test" / tid
                )["arcs"].to_numpy(int)

                ids = gp.sort_values(
                    "observation"
                )[col].astype(int).to_numpy()

                continuity.extend([
                    int(aa[a, 1] == aa[b, 0])
                    for a, b in zip(ids[:-1], ids[1:])
                    if 0 <= a < len(aa)
                    and 0 <= b < len(aa)
                ])

            hcomp.append({
                "method": col,
                "top1_accuracy": accuracy_score(
                    hmmdf.reference_arc,
                    hmmdf[col]
                ),
                "macro_f1": f1_score(
                    hmmdf.reference_arc,
                    hmmdf[col],
                    average="macro",
                    zero_division=0
                ),
                "route_continuity": (
                    float(np.mean(continuity))
                    if continuity else 0.0
                )
            })

        pd.DataFrame(hcomp).to_csv(
            out / "hmm_comparison.csv",
            index=False
        )

    # Save selected model information.
    with open(out / "selected_model.json", "w") as f:
        json.dump(
            {"selected_model": best_name},
            f,
            indent=2
        )

    print(table.to_string(index=False))
    print("Selected:", best_name)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()

    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw")
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path("results")
    )

    p.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts")
    )

    a = p.parse_args()

    main(a.data, a.out, a.artifacts)
