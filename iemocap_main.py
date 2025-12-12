#!/usr/bin/env python3
"""
IEMOCAP rude-trait pipeline (index -> template -> build_vector -> score_all -> plots)
"""

import os, re, sys, argparse, random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

import soundfile as sf
import librosa

# CONFIG
TARGET_SR = 16000
MODEL_NAME = "facebook/wav2vec2-base"
OUTPUT_DIR = "outputs_rude_iemocap"
CACHE_PATH = os.path.join(OUTPUT_DIR, "emb_cache.pt")     
RUDE_VECTOR_PATH = os.path.join(OUTPUT_DIR, "rude_vector.pt")
NEGATIVE_MULTIPLIER = 1.5
SEED = 42

EMO_MAP = {
    "neu":"neutral","hap":"happy","ang":"anger","sad":"sad",
    "exc":"excited","fru":"frustration","fea":"fear",
    "sur":"surprise","dis":"disgust"
}

# Utils

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

def load_wav(path, target_sr: int = TARGET_SR) -> torch.Tensor:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim == 2: y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return torch.from_numpy(np.ascontiguousarray(y))


# IEMOCAP indexing 

def _flatten_dialog_id(utt_id: str) -> str:
    parts = utt_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 3 else utt_id

def _try_wav_paths(wav_dir: Path, utt_id: str) -> Path | None:
    """
    Try to resolve WAV for both your flat layout and the nested layout.  
    """
    dialog = _flatten_dialog_id(utt_id)

    p = wav_dir / f"{utt_id}.wav"
    if p.exists(): return p

    p = wav_dir / f"{dialog}.wav"
    if p.exists(): return p

    p = wav_dir / dialog / f"{utt_id}.wav"
    if p.exists(): return p

    p = wav_dir / dialog / f"{dialog}.wav"
    if p.exists(): return p

    return None

def index_iemocap(root: str) -> pd.DataFrame:
    rows = []
    for sdir in sorted(Path(root).glob("Session*")):
        emo_dir = sdir / "dialog" / "EmoEvaluation"
        wav_dir = sdir / "dialog" / "wav"
        if not emo_dir.exists() or not wav_dir.exists():
            continue

        for txt in sorted(emo_dir.glob("*.txt")):
            with open(txt, "r", errors="ignore") as f:
                for line in f:
                    m = re.match(r"\[(.*?)\s*-\s*(.*?)\]\s+(\S+)\s+(\w+)", line.strip())
                    if not m:
                        continue
                    start, end, utt_id, emo = m.groups()
                    emo = EMO_MAP.get(emo.lower(), emo.lower())

                    wav_path = _try_wav_paths(wav_dir, utt_id)
                    if wav_path is None:
                        continue

                    rows.append({
                        "path": str(wav_path),
                        "filename": wav_path.name,
                        "session": sdir.name,
                        "dialog": _flatten_dialog_id(utt_id),
                        "start": start,
                        "end": end,
                        "emotion": emo
                    })

    df = pd.DataFrame(rows).drop_duplicates(subset=["path"]).reset_index(drop=True)
    return df


# Embeddings
class W2V2Embedder:
    def __init__(self, model_name=MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed_file(self, path: str) -> torch.Tensor:
        wav = load_wav(path)
        inp = self.processor(wav.numpy(), sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        inp = {k:v.to(self.device) for k,v in inp.items()}
        out = self.model(**inp)
        return out.last_hidden_state.mean(dim=1).squeeze(0).cpu()  # [H]

def load_or_init_cache():
    return torch.load(CACHE_PATH) if os.path.exists(CACHE_PATH) else {}

def save_cache(cache: dict):
    torch.save(cache, CACHE_PATH)

def get_embeddings(paths, embedder, cache):
    embs, updated = [], False
    for p in tqdm(paths, desc="Embedding"):
        key = Path(p).name 
        if key in cache and isinstance(cache[key], torch.Tensor):
            emb = cache[key]
        else:
            if not os.path.exists(p):
                continue
            emb = embedder.embed_file(p)
            cache[key] = emb; updated = True
        embs.append(emb)
    if updated: save_cache(cache)
    if not embs:
        raise ValueError("No embeddings computed.")
    return torch.stack(embs)


# Math helpers

def difference_in_means(pos, neg):
    return pos.mean(dim=0) - neg.mean(dim=0)

def cosine_projection(embs, v):
    v = v / (v.norm(p=2) + 1e-8)
    embs = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
    return (embs * v).sum(dim=1)


def cmd_index(args):
    ensure_dirs()
    df = index_iemocap(args.root)
    out_csv = args.out or os.path.join(OUTPUT_DIR, "iemocap_paths.csv")
    df.to_csv(out_csv, index=False)
    print(f"[index] Saved {len(df)} rows -> {out_csv}")
    if not df.empty and "emotion" in df.columns:
        print(df["emotion"].value_counts())

def cmd_template(args):
    df = pd.read_csv(args.paths_csv)
    emo = df["emotion"].str.lower()
    mask = emo.isin(["anger","frustration","ang","fru","annoyed","annoyance"])
    cand = df[mask].copy()
    if args.max_n: cand = cand.sample(min(args.max_n, len(cand)), random_state=SEED)
    out = cand[["path"]].rename(columns={"path":"filename"})
    out["0/1"] = 0
    out.to_csv(args.out, index=False)
    print(f"[template] Wrote {len(out)} candidates -> {args.out}")

def _resolve_label_paths(rude_csv: str, df_paths: pd.DataFrame):
    name2path = {Path(p).name: p for p in df_paths["path"].tolist()}
    df = pd.read_csv(rude_csv)
    if "filename" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path":"filename"})
    if "filename" not in df.columns or "0/1" not in df.columns:
        raise ValueError("rude CSV must have columns: filename (or path), 0/1")
    df["filename"] = df["filename"].astype(str)
    df["0/1"] = df["0/1"].astype(int)
    pos_paths, neg_paths = [], []
    for _, r in df.iterrows():
        v = r["filename"]
        full = v if os.path.exists(v) else name2path.get(Path(v).name)
        if not full: continue
        (pos_paths if r["0/1"] == 1 else neg_paths).append(full)
    return pos_paths, neg_paths

def cmd_build_vector(args):
    ensure_dirs(); set_seed()
    df_paths = pd.read_csv(args.paths_csv)
    pos_paths, neg_paths = _resolve_label_paths(args.rude_csv, df_paths)
    random.shuffle(neg_paths)
    neg_paths = neg_paths[:int(len(pos_paths) * args.neg_mult)]
    print(f"[build_vector] Pos={len(pos_paths)}  Neg={len(neg_paths)}")

    emb = W2V2Embedder(MODEL_NAME)
    cache = load_or_init_cache()

    pos_embs = get_embeddings(pos_paths, emb, cache)
    neg_embs = get_embeddings(neg_paths, emb, cache)

    rude_vec = difference_in_means(pos_embs, neg_embs)
    torch.save(rude_vec, RUDE_VECTOR_PATH)
    print(f"[build_vector] Saved rude vector -> {RUDE_VECTOR_PATH}")

def cmd_score_all(args):
    ensure_dirs()
    df_paths = pd.read_csv(args.paths_csv)
    rude_vec = torch.load(RUDE_VECTOR_PATH)
    emb = W2V2Embedder(MODEL_NAME)
    cache = load_or_init_cache()

    all_paths = df_paths["path"].tolist()
    all_embs = get_embeddings(all_paths, emb, cache)
    scores = cosine_projection(all_embs, rude_vec)

    out = df_paths.copy()
    out["rude_score"] = scores.tolist()
    out_csv = os.path.join(OUTPUT_DIR, "rude_scores_all_audio.csv")
    out.to_csv(out_csv, index=False)
    print(f"[score_all] Saved -> {out_csv}")

def cmd_plot_umap(args):
    import matplotlib.pyplot as plt
    import umap
    df_paths = pd.read_csv(args.paths_csv)

    label_map = {}
    if args.labels_csv and os.path.exists(args.labels_csv):
        df_r = pd.read_csv(args.labels_csv)
        if "path" in df_r.columns and "filename" not in df_r.columns:
            df_r = df_r.rename(columns={"path":"filename"})
        for _, r in df_r.iterrows():
            label_map[Path(str(r["filename"])).name] = int(r["0/1"])

    emb = W2V2Embedder(MODEL_NAME)
    cache = load_or_init_cache()

    paths = df_paths["path"].tolist()
    if args.max_n and len(paths) > args.max_n:
        paths = random.sample(paths, args.max_n)
    embs = get_embeddings(paths, emb, cache).numpy()
    names = [Path(p).name for p in paths]
    labels = np.array([label_map.get(n, 0) for n in names])

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED)
    embs2 = reducer.fit_transform(embs)

    plt.figure(figsize=(10,6))
    plt.scatter(embs2[labels==0,0], embs2[labels==0,1], c="royalblue", alpha=0.35, label="not rude")
    if (labels==1).any():
        plt.scatter(embs2[labels==1,0], embs2[labels==1,1], c="crimson", alpha=0.9, label="rude")
    plt.legend(); plt.title("UMAP: rude vs not-rude (IEMOCAP)")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.grid(alpha=0.2)
    plt.tight_layout(); plt.show()

def _prosody(path):
    y, sr = librosa.load(path, sr=TARGET_SR)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    return {
        "centroid_mean": float(np.nanmean(centroid)),
        "rms_mean": float(np.mean(rms)),
    }

def cmd_plot_prosody(args):
    import matplotlib.pyplot as plt
    df_paths = pd.read_csv(args.paths_csv)
    df_r = pd.read_csv(args.labels_csv)
    if "path" in df_r.columns and "filename" not in df_r.columns:
        df_r = df_r.rename(columns={"path":"filename"})
    name2path = {Path(p).name: p for p in df_paths["path"].tolist()}

    rows = []
    for _, r in df_r.iterrows():
        fname = Path(str(r["filename"])).name
        p = r["filename"] if os.path.exists(str(r["filename"])) else name2path.get(fname)
        if not p: continue
        feats = _prosody(p); feats["label"] = int(r["0/1"]); rows.append(feats)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No prosody rows; check labels/paths."); return

    m0 = df["label"]==0; m1 = df["label"]==1
    plt.figure(figsize=(10,6))
    plt.scatter(df.loc[m0,"centroid_mean"], df.loc[m0,"rms_mean"], c="royalblue", alpha=0.4, label="not rude")
    plt.scatter(df.loc[m1,"centroid_mean"], df.loc[m1,"rms_mean"], c="crimson",   alpha=0.9, label="rude")
    plt.xlabel("Spectral Centroid (harshness)"); plt.ylabel("RMS Energy")
    plt.title("Prosody space: rude vs not-rude")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()




def build_cli():
    p = argparse.ArgumentParser(description="IEMOCAP rude-trait pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="Index IEMOCAP and export utterance CSV")
    p_idx.add_argument("--root", required=True, help="Path to IEMOCAP_full_release")
    p_idx.add_argument("-o", "--out", default=None, help="Output CSV path")
    p_idx.set_defaults(func=cmd_index)

    p_tmp = sub.add_parser("template", help="Create labeling template from anger/frustration")
    p_tmp.add_argument("--paths_csv", default=os.path.join(OUTPUT_DIR, "iemocap_paths.csv"))
    p_tmp.add_argument("-o", "--out", default="rude_trait.csv")
    p_tmp.add_argument("--max_n", type=int, default=400, help="Max rows in template")
    p_tmp.set_defaults(func=cmd_template)

    p_vec = sub.add_parser("build_vector", help="Build rude vector from labeled CSV")
    p_vec.add_argument("--paths_csv", default=os.path.join(OUTPUT_DIR, "iemocap_paths.csv"))
    p_vec.add_argument("--rude_csv", required=True)
    p_vec.add_argument("--neg_mult", type=float, default=NEGATIVE_MULTIPLIER)
    p_vec.set_defaults(func=cmd_build_vector)

    p_score = sub.add_parser("score_all", help="Score all utterances using the rude vector")
    p_score.add_argument("--paths_csv", default=os.path.join(OUTPUT_DIR, "iemocap_paths.csv"))
    p_score.set_defaults(func=cmd_score_all)

    p_umap = sub.add_parser("plot_umap", help="UMAP scatter of rude vs not-rude")
    p_umap.add_argument("--paths_csv", default=os.path.join(OUTPUT_DIR, "iemocap_paths.csv"))
    p_umap.add_argument("--labels_csv", default="rude_trait.csv")
    p_umap.add_argument("--max_n", type=int, default=2500)
    p_umap.set_defaults(func=cmd_plot_umap)

    p_pros = sub.add_parser("plot_prosody", help="Prosody scatter (centroid vs RMS)")
    p_pros.add_argument("--paths_csv", default=os.path.join(OUTPUT_DIR, "iemocap_paths.csv"))
    p_pros.add_argument("--labels_csv", default="rude_trait.csv")
    p_pros.set_defaults(func=cmd_plot_prosody)

    return p

def main():
    set_seed(); ensure_dirs()
    cli = build_cli()
    args = cli.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
