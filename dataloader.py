from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent / "data"


def load_data(subject="both"):
    """Load one course or concatenate Mathematics and Portuguese data."""
    paths = {
        "mat": DATA_DIR / "student-mat.csv",
        "por": DATA_DIR / "student-por.csv",
    }
    if subject not in {"mat", "por", "both"}:
        raise ValueError("subject must be 'mat', 'por', or 'both'")

    names = ["mat", "por"] if subject == "both" else [subject]
    frames = []
    for name in names:
        frame = pd.read_csv(paths[name], sep=";")
        frame["course"] = name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    y = (data.pop("G3") >= 10).astype(int)
    data = data.drop(columns=["G1", "G2", "G3"], errors="ignore")
    return data, y
