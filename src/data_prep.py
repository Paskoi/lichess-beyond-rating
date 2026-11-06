from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "games.parquet"
CLEAN = ROOT / "data" / "clean.parquet"

SENTINELS = ["?", "-"]
CATEGORICAL = ["Result", "Termination", "event_type", "TimeControl", "ECO", "Opening"]

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    text_cols = df.select_dtypes(include=["object", "string"]).columns
    df[text_cols] = df[text_cols].replace(SENTINELS, pd.NA)

    df["event_type"] = df["Event"].str.replace(r"\s*https?://\S+", "", regex=True).str.strip()

    rules = {
        "no_result": df["Result"] == "*",
        "abandoned": df["Termination"] == "Abandoned",
        "rules_infraction": df["Termination"] == "Rules infraction",
        "unterminated": df["Termination"] == "Unterminated",
        "correspondence": df["event_type"] == "Rated Correspondence game",
    }
    summary = pd.Series({k: int(m.sum()) for k, m in rules.items()}, name="rows_flagged")
    df = df[~pd.concat(rules, axis=1).any(axis=1)].copy()

    elo_cols = ["WhiteElo", "BlackElo"]
    # Unparsable value should fail, not become NaN
    df[elo_cols] = df[elo_cols].apply(pd.to_numeric).astype("Int16")

    df["utc_dt"] = pd.to_datetime(df["UTCDate"] + " " + df["UTCTime"], format="%Y.%m.%d %H:%M:%S")
    # the game id identifies a game
    df["game_id"] = df["Site"].str.rsplit("/", n=1).str[-1]

    df = df.drop(columns=["Event", "Site", "UTCDate", "UTCTime"])
    df[CATEGORICAL] = df[CATEGORICAL].astype("category")

    return df.reset_index(drop=True), summary

if __name__ == "__main__":
    df_clean, summary = clean(pd.read_parquet(RAW))
    print(summary)
    print(f"Clean dataset: {len(df_clean):,} games")
    df_clean.to_parquet(CLEAN, index=False)
    print(f"Saved to {CLEAN}")