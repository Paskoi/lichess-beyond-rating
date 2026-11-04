import argparse
import io
import re
from pathlib import Path

import polars as pl
import zstandard as zstd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "data" / "lichess_db_standard_rated_2017-01.pgn.zst"
DEFAULT_OUT = ROOT / "data" / "raw"

TAG = re.compile(r'\[(\w+) "(.*)"\]')
KEEP = ["Event", "Site", "White", "Black", "Result", "WhiteElo", "BlackElo",
        "TimeControl", "ECO", "Opening", "Termination", "UTCDate", "UTCTime"]


def open_zst(path):
    raw = open(path, "rb")
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    return io.TextIOWrapper(dctx.stream_reader(raw), encoding="utf-8")


def games(stream, keep_moves):
    tags, moves, in_moves = {}, [], False
    for line in stream:
        if line.startswith("["):
            if in_moves:
                if keep_moves:
                    tags["moves"] = " ".join(moves)
                yield tags
                tags, moves, in_moves = {}, [], False
            m = TAG.match(line)
            if m and m[1] in KEEP:
                tags[m[1]] = m[2]
        elif line.strip():
            in_moves = True
            if keep_moves:
                moves.append(line.strip())
    if tags:
        if keep_moves:
            tags["moves"] = " ".join(moves)
        yield tags


def to_parquet(stream, out_dir, keep_moves, chunk=500_000, limit=None):
    schema = {k: pl.Utf8 for k in KEEP + (["moves"] if keep_moves else [])}
    buf, part, n = [], 0, 0

    def flush():
        nonlocal buf, part
        pl.from_dicts(buf, schema=schema).write_parquet(out_dir / f"part_{part:03}.parquet")
        print(f"  chunk {part:03} written, {n:,} games total")
        buf, part = [], part + 1

    for g in games(stream, keep_moves):
        buf.append(g)
        n += 1
        if len(buf) == chunk:
            flush()
        if limit and n >= limit:
            break
    if buf:
        flush()
    return n


def merge_parts(out_dir, target):
    pl.scan_parquet(out_dir / "part_*.parquet").sink_parquet(target)
    for p in out_dir.glob("part_*.parquet"):
        p.unlink()


def target_path(out_dir, keep_moves, limit):
    name = "games"
    if keep_moves:
        name += "_moves"
    if limit:
        name += f"_sample{limit}"
    return out_dir.parent / f"{name}.parquet"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", nargs="?", type=Path, default=DEFAULT_SRC)
    ap.add_argument("out_dir", nargs="?", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="max number of games (for testing)")
    ap.add_argument("--moves", action="store_true", help="also store raw move text")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for p in args.out_dir.glob("part_*.parquet"):
        p.unlink()

    n = to_parquet(open_zst(args.src), args.out_dir, args.moves, limit=args.limit)
    target = target_path(args.out_dir, args.moves, args.limit)
    merge_parts(args.out_dir, target)
    print(f"Done: {n:,} games written to {target}")