import pandas as pd

STARTING_RATING = 1500 # rating = 1500 often means first rated game in that speed
VALID_START = pd.Timestamp("2017-01-18")
TEST_START = pd.Timestamp("2017-01-25")
RESULT_MAP = {"0-1": 0, "1/2-1/2": 1, "1-0": 2} # Black win | draw | White win
ECO_MIN_GAMES = 500


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    white = df["WhiteElo"].astype(int)
    black = df["BlackElo"].astype(int)
    out["elo_diff"] = white - black
    out["elo_avg"] = (white + black) / 2
    out["white_provisional"] = (white == STARTING_RATING).astype(int)
    out["black_provisional"] = (black == STARTING_RATING).astype(int)

    tc = df["TimeControl"].astype(str).str.split("+", expand=True).astype(int)
    out["base_time"] = tc[0]
    out["increment"] = tc[1]

    event = df["event_type"].astype(str)
    out["speed"] = event.str.split().str[1].astype("category")
    out["is_tournament"] = event.str.contains("tournament").astype(int)

    out["eco"] = df["ECO"].astype("str").fillna("unknown")

    out["result"] = df["Result"].astype(str).map(RESULT_MAP)
    out["time_forfeit"] = (df["Termination"] == "Time forfeit").astype(int)
    out["utc_dt"] = df["utc_dt"]
    return out


def time_split(feat: pd.DataFrame):
    train = feat[feat["utc_dt"] < VALID_START].copy()
    valid = feat[(feat["utc_dt"] >= VALID_START) & (feat["utc_dt"] < TEST_START)].copy()
    test = feat[feat["utc_dt"] >= TEST_START].copy()
    return train, valid, test


def fit_eco_vocab(train_eco: pd.Series, min_games: int = ECO_MIN_GAMES) -> set:
    counts = train_eco.value_counts()
    return set(counts[counts >= min_games].index) | {"unknown"}


def apply_eco_vocab(eco: pd.Series, vocab: set) -> pd.Series:
    categories = sorted(vocab | {"other"})
    return pd.Series(
        pd.Categorical(eco.where(eco.isin(vocab), "other"), categories=categories),
        index=eco.index,
    )

def calibration_table(proba_class, y, class_idx, n_bins=10):
    t = pd.DataFrame({"pred": proba_class, "observed": (y == class_idx)})
    t["bin"] = (t["pred"] * n_bins).astype(int).clip(upper=n_bins - 1)   # 0 = 0–10%, 1 = 10–20%, ..., 9 = 90–100%
    return t.groupby("bin")[["pred", "observed"]].mean()