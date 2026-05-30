"""
STEP 1 — DATASET PREPARATION
=============================
Run this locally or in a Kaggle notebook cell before fine-tuning.
Converts train.csv into a JSONL fine-tuning dataset and saves a
validation split using day-49 rows only.

Output files (upload these to your Kaggle Dataset):
  - finetune_train.jsonl
  - finetune_val.jsonl
  - test_prompts.jsonl
"""

import csv
import json
import random
import os

random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────────
TRAIN_CSV       = "train.csv"        # adjust paths as needed
TEST_CSV        = "test.csv"
OUT_DIR         = "."                # where to write the jsonl files
DAY49_OVERSAMPLE = 4                 # repeat day-49 rows this many times
# ────────────────────────────────────────────────────────────────────────────


def fmt_timestamp(ts: str) -> str:
    """'10:0' -> '10:00', '2:15' -> '02:15'"""
    h, m = ts.split(":")
    return f"{int(h):02d}:{int(m):02d}"


def row_to_prompt(row: dict, include_demand: bool = True) -> dict:
    """
    Serialize one CSV row into a chat-style prompt/completion pair.
    The model sees all features and must predict a float in [0, 1].
    """
    road      = row.get("RoadType", "").strip()   or "Unknown"
    lanes     = row.get("NumberofLanes", "").strip() or "Unknown"
    large_v   = row.get("LargeVehicles", "").strip() or "Unknown"
    landmarks = row.get("Landmarks", "").strip()  or "Unknown"
    temp      = row.get("Temperature", "").strip()
    weather   = row.get("Weather", "").strip()    or "Unknown"
    day       = row.get("day", "").strip()
    ts        = fmt_timestamp(row.get("timestamp", "0:0"))
    geohash   = row.get("geohash", "").strip()

    temp_str = f"{float(temp):.1f}°C" if temp else "Unknown"

    user_msg = (
        f"Predict the traffic demand (a value between 0.0 and 1.0) "
        f"for the following road segment.\n\n"
        f"Location: geohash={geohash}\n"
        f"Day: {day}, Time: {ts}\n"
        f"Road type: {road}, Lanes: {lanes}\n"
        f"Large vehicles allowed: {large_v}\n"
        f"Landmarks nearby: {landmarks}\n"
        f"Temperature: {temp_str}, Weather: {weather}\n\n"
        f"Respond with ONLY a single decimal number between 0.0000 and 1.0000. "
        f"No explanation, no units, just the number."
    )

    if include_demand:
        demand = float(row["demand"])
        assistant_msg = f"{demand:.6f}"
        return {
            "messages": [
                {"role": "user",    "content": user_msg},
                {"role": "assistant","content": assistant_msg},
            ]
        }
    else:
        # For inference — no assistant turn
        return {
            "index":  row["Index"],
            "prompt": user_msg,
        }


def main():
    print("Loading train.csv ...")
    with open(TRAIN_CSV) as f:
        train_rows = list(csv.DictReader(f))

    print("Loading test.csv ...")
    with open(TEST_CSV) as f:
        test_rows = list(csv.DictReader(f))

    # ── Split: day-49 rows become validation, rest is fine-tune train ────────
    d48_rows = [r for r in train_rows if r["day"] == "48"]
    d49_rows = [r for r in train_rows if r["day"] == "49"]

    print(f"Day-48 rows: {len(d48_rows)}")
    print(f"Day-49 rows: {len(d49_rows)}")

    # Use 80 % of day-49 for val, keep 20 % in train too (they're precious)
    random.shuffle(d49_rows)
    val_cutoff   = int(len(d49_rows) * 0.8)
    d49_val      = d49_rows[:val_cutoff]
    d49_train    = d49_rows[val_cutoff:]

    # Build fine-tune split: all day-48 + day-49 train oversampled
    finetune_rows = d48_rows + d49_train * DAY49_OVERSAMPLE
    random.shuffle(finetune_rows)

    print(f"Fine-tune rows (after oversampling): {len(finetune_rows)}")
    print(f"Validation rows:                     {len(d49_val)}")

    # ── Write JSONL ──────────────────────────────────────────────────────────
    ft_path  = os.path.join(OUT_DIR, "finetune_train.jsonl")
    val_path = os.path.join(OUT_DIR, "finetune_val.jsonl")
    tst_path = os.path.join(OUT_DIR, "test_prompts.jsonl")

    with open(ft_path, "w") as f:
        for row in finetune_rows:
            f.write(json.dumps(row_to_prompt(row, include_demand=True)) + "\n")
    print(f"Wrote {ft_path}")

    with open(val_path, "w") as f:
        for row in d49_val:
            f.write(json.dumps(row_to_prompt(row, include_demand=True)) + "\n")
    print(f"Wrote {val_path}")

    with open(tst_path, "w") as f:
        for row in test_rows:
            f.write(json.dumps(row_to_prompt(row, include_demand=False)) + "\n")
    print(f"Wrote {tst_path}")

    # ── Quick sanity check ───────────────────────────────────────────────────
    with open(ft_path) as f:
        sample = json.loads(f.readline())
    print("\n─── SAMPLE PROMPT ───")
    print(sample["messages"][0]["content"])
    print("─── SAMPLE TARGET ───")
    print(sample["messages"][1]["content"])


if __name__ == "__main__":
    main()