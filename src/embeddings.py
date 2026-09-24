import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def load_esm_model(model_name="facebook/esm2_t12_35M_UR50D", device="cpu"):
    """
    Load frozen ESM-2 model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return tokenizer, model


def load_rita_model(model_name="lightonai/RITA_s", device="cpu"):
    """
    Load frozen RITA-s model and tokenizer with compatibility patch for transformers >= 4.38.
    """
    # Ensure all_tied_weights_keys exists on dynamic class to prevent AttributeError
    model_class = get_class_from_dynamic_module(
        f"{model_name}--rita_modeling.RITAModelForCausalLM", model_name
    )
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    # Convert to float32 for stable CPU/GPU inference
    model = model.float()
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return tokenizer, model


def extract_esm_embeddings(df, tokenizer, model, device="cpu", batch_size=16):
    """
    Extract mutation-position embedding from ESM-2.
    Token indexing:
      Index 0 is <cls> token.
      Biological position pos (1-based) corresponds to token index pos.
    """
    embeddings = []
    sequences = df["mutated_sequence"].tolist()
    positions = df["pos"].tolist()

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        batch_pos = positions[i : i + batch_size]

        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # shape: (batch, seq_len, 480)

        for b, pos in enumerate(batch_pos):
            # pos is 1-based, index pos corresponds to the mutated residue
            token_idx = pos
            emb = last_hidden[b, token_idx, :].cpu().numpy()
            embeddings.append(emb)

    return np.array(embeddings, dtype=np.float32)


def extract_rita_embeddings(df, tokenizer, model, device="cpu", batch_size=1):
    """
    Extract mutation-position embedding from RITA-s sequence by sequence.
    Token indexing:
      RITA does not prepend a <cls> token.
      Biological position pos (1-based) corresponds to token index pos - 1.
      RITA vocab has size 26 (indices 0..25); index 26 is redundant 'E' (token 7 is primary 'E').
    """
    embeddings = []
    sequences = df["mutated_sequence"].tolist()
    positions = df["pos"].tolist()

    for idx, (seq, pos) in enumerate(zip(sequences, positions)):
        inputs = tokenizer(seq, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["input_ids"] = torch.clamp(inputs["input_ids"], 0, 25)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            if hidden.ndim == 3:
                hidden = hidden[0]

        token_idx = pos - 1
        emb = hidden[token_idx, :].cpu().numpy()
        embeddings.append(emb)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(sequences):
            print(f"  RITA progress: {idx + 1}/{len(sequences)} extracted")

    return np.array(embeddings, dtype=np.float32)


def get_or_create_cached_embeddings(
    df_train, df_val, df_test, cache_dir, device="cpu", batch_size=16
):
    """
    Load cached embeddings if available, otherwise compute and save them.
    Cached files:
      esm_train.npy, esm_val.npy, esm_test.npy
      source_train.npy, source_val.npy, source_test.npy
    """
    os.makedirs(cache_dir, exist_ok=True)

    esm_tr_path = os.path.join(cache_dir, "esm_train.npy")
    esm_val_path = os.path.join(cache_dir, "esm_validation.npy")
    esm_test_path = os.path.join(cache_dir, "esm_test.npy")

    src_tr_path = os.path.join(cache_dir, "source_train.npy")
    src_val_path = os.path.join(cache_dir, "source_validation.npy")
    src_test_path = os.path.join(cache_dir, "source_test.npy")

    all_exist = all(
        os.path.exists(p)
        for p in [
            esm_tr_path,
            esm_val_path,
            esm_test_path,
            src_tr_path,
            src_val_path,
            src_test_path,
        ]
    )

    if all_exist:
        print(f"Loading cached embeddings from {cache_dir}...")
        return {
            "esm_train": np.load(esm_tr_path),
            "esm_val": np.load(esm_val_path),
            "esm_test": np.load(esm_test_path),
            "source_train": np.load(src_tr_path),
            "source_val": np.load(src_val_path),
            "source_test": np.load(src_test_path),
        }

    print("Cached embeddings not found or incomplete. Computing from scratch...")

    esm_exist = all(os.path.exists(p) for p in [esm_tr_path, esm_val_path, esm_test_path])
    if esm_exist:
        print("ESM-2 embeddings already exist on disk, loading them...")
        esm_train = np.load(esm_tr_path)
        esm_val = np.load(esm_val_path)
        esm_test = np.load(esm_test_path)
    else:
        print("1/2: Extracting ESM-2 35M embeddings...")
        esm_tok, esm_mod = load_esm_model(device=device)
        esm_train = extract_esm_embeddings(df_train, esm_tok, esm_mod, device, batch_size)
        esm_val = extract_esm_embeddings(df_val, esm_tok, esm_mod, device, batch_size)
        esm_test = extract_esm_embeddings(df_test, esm_tok, esm_mod, device, batch_size)

        np.save(esm_tr_path, esm_train)
        np.save(esm_val_path, esm_val)
        np.save(esm_test_path, esm_test)
        del esm_mod, esm_tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("2/2: Extracting RITA-s embeddings...")
    rita_tok, rita_mod = load_rita_model(device=device)
    src_train = extract_rita_embeddings(df_train, rita_tok, rita_mod, device, batch_size)
    src_val = extract_rita_embeddings(df_val, rita_tok, rita_mod, device, batch_size)
    src_test = extract_rita_embeddings(df_test, rita_tok, rita_mod, device, batch_size)

    np.save(src_tr_path, src_train)
    np.save(src_val_path, src_val)
    np.save(src_test_path, src_test)
    del rita_mod, rita_tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Embeddings successfully computed and cached!")
    return {
        "esm_train": esm_train,
        "esm_val": esm_val,
        "esm_test": esm_test,
        "source_train": src_train,
        "source_val": src_val,
        "source_test": src_test,
    }
