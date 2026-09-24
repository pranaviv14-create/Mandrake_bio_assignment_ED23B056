import os
import re
import pandas as pd


def parse_mutant(mutant_str):
    """
    Parse mutation string like 'A110D' into (wt_aa, position, mut_aa).
    Position is 1-based biological residue index.
    """
    match = re.match(r"^([A-Z])(\d+)([A-Z])$", mutant_str.strip())
    if not match:
        raise ValueError(f"Invalid mutant format: {mutant_str}")
    wt_aa = match.group(1)
    pos = int(match.group(2))
    mut_aa = match.group(3)
    return wt_aa, pos, mut_aa


def load_dataset(data_dir):
    """
    Load train, validation, and test datasets and attach parsed position.
    """
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")
    fasta_path = os.path.join(data_dir, "reference.fasta")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    for df in [df_train, df_val, df_test]:
        parsed = df["mutant"].apply(parse_mutant)
        df["wt_aa"] = [p[0] for p in parsed]
        df["pos"] = [p[1] for p in parsed]
        df["mut_aa"] = [p[2] for p in parsed]

    ref_seq = ""
    if os.path.exists(fasta_path):
        with open(fasta_path, "r") as f:
            for line in f:
                if not line.startswith(">"):
                    ref_seq += line.strip()

    return df_train, df_val, df_test, ref_seq


def verify_split_leakage(df_train, df_val, df_test):
    """
    Verify that there is ZERO overlap in mutated residue positions across splits.
    """
    train_pos = set(df_train["pos"])
    val_pos = set(df_val["pos"])
    test_pos = set(df_test["pos"])

    train_val_overlap = train_pos.intersection(val_pos)
    train_test_overlap = train_pos.intersection(test_pos)
    val_test_overlap = val_pos.intersection(test_pos)

    assert len(train_val_overlap) == 0, f"Leakage: Train/Val overlap on positions: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Leakage: Train/Test overlap on positions: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Leakage: Val/Test overlap on positions: {val_test_overlap}"

    return {
        "train_variants": len(df_train),
        "val_variants": len(df_val),
        "test_variants": len(df_test),
        "train_positions": (min(train_pos), max(train_pos), len(train_pos)),
        "val_positions": (min(val_pos), max(val_pos), len(val_pos)),
        "test_positions": (min(test_pos), max(test_pos), len(test_pos)),
    }
