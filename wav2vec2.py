#!/usr/bin/env python3
# wav2vec2_ser.py
# Python 3.10+  |  pip install torch torchaudio transformers datasets evaluate numpy pandas soundfile

import os, re, glob, json, random, shutil, argparse, collections, inspect
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import torch
import torchaudio
import soundfile as sf
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

try:
    import pandas as pd
except Exception:
    pd = None

import transformers
from transformers import (
    AutoProcessor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)

print("HF transformers version:", transformers.__version__)
print("TrainingArguments comes from:", inspect.getfile(TrainingArguments))

# ============== Filename parsing: emotion after speaker code (ANG, DIS, FEA, HAP, NEU, SAD) ==============
EMO_TAGS = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
EMO_RE = re.compile(r"_(ANG|DIS|FEA|HAP|NEU|SAD)(?:_|$)")  # match _NEU_ or end-of-stem

def is_real_wav(name: str) -> bool:
    base = os.path.basename(name)
    return base.lower().endswith(".wav") and not base.startswith("._")  # ignore macOS resource forks

def extract_label_from_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path.strip()))[0]
    m = EMO_RE.search(stem)
    if m:
        return EMO_TAGS[m.group(1)]
    # fallback: first underscore token that matches
    for tok in stem.split("_"):
        if tok in EMO_TAGS:
            return EMO_TAGS[tok]
    raise ValueError(f"Could not find an emotion tag in filename: {os.path.basename(path)}")

def list_wavs_flat(root: str) -> List[str]:
    if not os.path.isdir(root): return []
    return [os.path.join(root, f) for f in os.listdir(root) if is_real_wav(f)]

def list_wavs_recursive(root: str) -> List[str]:
    return [p for p in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True) if is_real_wav(p)]

# ============== Inspect flat folder by EMOTION ==============
def inspect_source_dir(source_dir: str, max_examples_per_label: int = 8):
    wavs = list_wavs_flat(source_dir)
    if not wavs:
        raise RuntimeError(f"No .wav files found in: {source_dir}")

    by_label, miss = collections.defaultdict(list), []
    for p in wavs:
        try:
            lab = extract_label_from_name(p)
            by_label[lab].append(p)
        except Exception:
            miss.append(p)

    print("\n=== Dataset Inspection (emotion-level) ===")
    print(f"Total WAV files: {len(wavs)}")
    print(f"Detected emotion labels ({len(by_label)}): {sorted(by_label.keys())}\n")
    print("Counts per emotion:")
    for lab in sorted(by_label):
        print(f"  {lab:10s} {len(by_label[lab])}")
    if miss:
        print(f"\n⚠ Could not parse emotion tag for {len(miss)} files. Example:")
        print(" ", os.path.basename(miss[0]))

    print("\nExamples:")
    for lab in sorted(by_label):
        ex = [os.path.basename(x) for x in sorted(by_label[lab])[:max_examples_per_label]]
        print(f"  {lab:10s} -> {ex}")
    print("==========================\n")

# ============== Split flat folder -> train/val/test (per-emotion robust) ==============
def split_dataset(
    source_dir: str,
    target_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    make_metadata_csv: bool = True,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    os.makedirs(target_dir, exist_ok=True)
    rnd = random.Random(seed)

    wavs = list_wavs_flat(source_dir)
    if not wavs:
        raise RuntimeError(f"No .wav files found in: {source_dir}")

    by_label: Dict[str, List[str]] = collections.defaultdict(list)
    skipped = 0
    for p in wavs:
        try:
            by_label[extract_label_from_name(p)].append(p)
        except Exception:
            skipped += 1
    if skipped:
        print(f"Note: {skipped} files skipped (no emotion tag).")

    train_files, val_files, test_files = [], [], []
    for lab, paths in sorted(by_label.items()):
        paths = sorted(paths)
        rnd.shuffle(paths)
        n = len(paths)
        if n == 1:
            train_files += paths; continue
        if n == 2:
            train_files.append(paths[0])
            (val_files if val_ratio > 0 else test_files).append(paths[1]); continue
        n_train = max(1, int(round(n * train_ratio)))
        n_val   = int(round(n * val_ratio))
        n_test  = n - n_train - n_val
        if n_test < 0:
            deficit = -n_test
            take = min(deficit, n_val)
            n_val -= take; n_test += take
        if n_test < 0:
            deficit = -n_test
            take = min(deficit, max(0, n_train - 1))
            n_train -= take; n_test += take
        if n_train < 1:
            n_train = 1
            if n_val > 0: n_val -= 1
            n_test = n - n_train - n_val

        train_files += paths[:n_train]
        val_files   += paths[n_train:n_train+n_val]
        test_files  += paths[n_train+n_val:]

    def copy_into(split_name: str, paths: List[str]):
        for p in paths:
            lab = extract_label_from_name(p)
            out_dir = os.path.join(target_dir, split_name, lab)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy(p, os.path.join(out_dir, os.path.basename(p)))

    copy_into("train", train_files)
    copy_into("val",   val_files)
    copy_into("test",  test_files)

    print(f"\nSplit sizes -> train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")
    print(f"Finished. Split dataset at: {target_dir}")

    if make_metadata_csv and pd is not None:
        def dump_csv(name, paths):
            labs = [extract_label_from_name(p) for p in paths]
            df = pd.DataFrame({"path": paths, "label": labs})
            csv_path = os.path.join(target_dir, f"{name}_metadata.csv")
            df.to_csv(csv_path, index=False); print(f"Wrote {csv_path}")
        dump_csv("train", train_files); dump_csv("val", val_files); dump_csv("test", test_files)

# ============== Dataset + Collator (uses soundfile to avoid torchcodec) ==============
class SERDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        items: List[Tuple[str, int]],
        processor,
        target_sr: int = 16000,
        max_seconds: float = 6.0,
        min_seconds: float = 0.2,
        trim_or_pad: bool = True,
    ):
        self.items = items
        self.processor = processor
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_seconds)
        self.min_len = int(target_sr * min_seconds)
        self.trim_or_pad = trim_or_pad

    def _load_wav(self, path: str) -> np.ndarray:
        # 1) robust read (no torchcodec)
        audio, sr = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
        # mono
        if audio.shape[1] > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        else:
            audio = audio[:, 0]
        # 2) resample with torchaudio (works without codec)
        if sr != self.target_sr:
            audio = torchaudio.transforms.Resample(sr, self.target_sr)(torch.from_numpy(audio)).numpy()
        # 3) trim/pad
        if self.trim_or_pad:
            if audio.shape[0] > self.max_len:
                audio = audio[:self.max_len]
            elif audio.shape[0] < self.max_len:
                audio = np.pad(audio, (0, self.max_len - audio.shape[0]), mode="constant")
        if audio.shape[0] < self.min_len:
            audio = np.pad(audio, (0, self.min_len - audio.shape[0]), mode="constant")
        return audio

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, label_id = self.items[idx]
        audio = self._load_wav(path)
        return {"input_values": audio, "labels": int(label_id), "path": path}

@dataclass
class DataCollatorWav2Vec2:
    processor: AutoProcessor
    sampling_rate: int = 16000
    def __call__(self, features):
        input_values = [f["input_values"] for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        batch = self.processor(input_values, sampling_rate=self.sampling_rate,
                               return_tensors="pt", padding=True)
        batch["labels"] = labels
        return batch

# ============== Class-weighted Trainer (no funky model subclass) ==============
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights  # torch.Tensor or None

    # Accept extra kwargs like num_items_in_batch from newer Trainer
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # Remove labels from inputs so base model doesn't compute loss internally
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**model_inputs)  # forward pass
        logits = outputs.get("logits")

        # class-weighted cross entropy
        cw = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ============== Helpers for split dirs ==============
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def resolve_label_name(id2label, idx: int):
    # Handles dict with int keys, dict with str keys, or list/tuple
    if isinstance(id2label, dict):
        if idx in id2label:
            return id2label[idx]
        if str(idx) in id2label:
            return id2label[str(idx)]
        # normalize keys once if needed
        norm = { (int(k) if isinstance(k, str) and k.isdigit() else k): v for k, v in id2label.items() }
        return norm.get(idx, f"label_{idx}")
    # list/tuple
    try:
        return id2label[idx]
    except Exception:
        return f"label_{idx}"

def get_label_names_from_train(train_dir: str) -> List[str]:
    labs = []
    for name in sorted(os.listdir(train_dir)):
        p = os.path.join(train_dir, name)
        if os.path.isdir(p) and list_wavs_recursive(p):
            labs.append(name.lower())
    if not labs:
        raise RuntimeError("No label subfolders found in train split.")
    return labs

def load_split_items(split_dir: str, label2id: Dict[str, int]) -> List[Tuple[str, int]]:
    items = []
    for lab in sorted(os.listdir(split_dir)):
        lab_dir = os.path.join(split_dir, lab)
        if not os.path.isdir(lab_dir): continue
        lab_norm = lab.lower()
        if lab_norm not in label2id: continue
        for p in list_wavs_recursive(lab_dir):
            items.append((p, label2id[lab_norm]))
    if not items:
        raise RuntimeError(f"No wavs found in: {split_dir}")
    return items

def compute_class_weights(train_items: List[Tuple[str, int]], num_labels: int) -> torch.Tensor:
    counts = np.zeros(num_labels, dtype=np.int64)
    for _, y in train_items: counts[y] += 1
    weights = 1.0 / np.maximum(counts, 1)
    weights = weights * (num_labels / weights.sum())
    print("Class counts:", counts.tolist())
    print("Class weights:", [round(float(w), 3) for w in weights])
    return torch.tensor(weights, dtype=torch.float32)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }

# ============== TrainingArguments (version-tolerant, and macOS-friendly) ==============
def make_training_args(
    out_dir,
    batch_size=8,
    epochs=10,
    lr=3e-5,
    weight_decay=0.01,
    logging_steps=50,
    num_workers=0,             # IMPORTANT on macOS/MPS: use 0 workers to avoid crashes
    seed=42,
):
    return TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=False,  # avoid MPS pin memory warnings
        report_to="none",
        seed=seed,
        # deliberately omit evaluation_strategy/save_strategy/warmup_ratio/etc. for compatibility
    )

# ============== Build model (base model; feature extractor frozen) ==============
def build_model(model_name: str, num_labels: int, id2label: Dict[int, str], label2id: Dict[str, int],
                freeze_feature_extractor: bool = True):
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )
    if freeze_feature_extractor:
        for p in model.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False
    return model

# ============== Train / Test / Predict ==============
def train_ser(
    data_root: str,
    model_name: str = "facebook/wav2vec2-base",
    out_dir: str = "./ser_wav2vec2_ckpt",
    seed: int = 42,
    sr_target: int = 16000,
    max_seconds: float = 6.0,
    batch_size: int = 8,
    epochs: int = 10,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    num_workers: int = 0,
    use_class_weights: bool = True,
):
    set_seed(seed)
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")

    labels = get_label_names_from_train(train_dir)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    print("Labels:", labels)

    processor = AutoProcessor.from_pretrained(model_name)
    train_items = load_split_items(train_dir, label2id)
    val_items   = load_split_items(val_dir, label2id)

    class_weights = compute_class_weights(train_items, num_labels=len(labels)) if use_class_weights else None

    train_ds = SERDataset(train_items, processor, target_sr=sr_target, max_seconds=max_seconds)
    val_ds   = SERDataset(val_items,   processor, target_sr=sr_target, max_seconds=max_seconds)
    collator = DataCollatorWav2Vec2(processor, sampling_rate=sr_target)

    model = build_model(
        model_name=model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        freeze_feature_extractor=True,
    )

    args = make_training_args(
        out_dir=out_dir,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,  # 0 on macOS/MPS
        seed=seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,  # (avoids "tokenizer deprecated" warning)
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)

    val_metrics = trainer.evaluate()
    print("\nValidation metrics:", json.dumps(val_metrics, indent=2))

def evaluate_on_test(
    data_root: str,
    ckpt_dir: str,
    sr_target: int = 16000,
    max_seconds: float = 6.0,
    batch_size: int = 8,
):
    train_dir = os.path.join(data_root, "train")
    test_dir  = os.path.join(data_root, "test")

    labels = get_label_names_from_train(train_dir)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    processor = AutoProcessor.from_pretrained(ckpt_dir)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(ckpt_dir).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_items = load_split_items(test_dir, label2id)
    test_ds = SERDataset(test_items, processor, target_sr=sr_target, max_seconds=max_seconds)
    collator = DataCollatorWav2Vec2(processor, sampling_rate=sr_target)

    dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     collate_fn=collator, num_workers=0, pin_memory=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in dl:
            labels_t = batch.pop("labels").numpy()
            for k in batch: batch[k] = batch[k].to(device)
            logits = model(**batch).logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=-1)
            y_true.append(labels_t); y_pred.append(preds)

    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    print("\nTEST RESULTS")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 (macro):", f1_score(y_true, y_pred, average="macro"))
    print("F1 (weighted):", f1_score(y_true, y_pred, average="weighted"))
    print("\nPer-class report:\n", classification_report(y_true, y_pred, target_names=labels))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
def resolve_label_name(id2label, idx: int):
    if isinstance(id2label, dict):
        if idx in id2label:
            return id2label[idx]
        if str(idx) in id2label:
            return id2label[str(idx)]
        # normalize keys
        norm = {
            (int(k) if isinstance(k, str) and k.isdigit() else k): v
            for k, v in id2label.items()
        }
        return norm.get(idx, f"label_{idx}")
    # list/tuple fallback
    try:
        return id2label[idx]
    except Exception:
        return f"label_{idx}"
    
def predict_wav(wav_path: str, ckpt_dir: str, sr_target: int = 16000):
    processor = AutoProcessor.from_pretrained(ckpt_dir)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(ckpt_dir).eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ---- load wav using safe soundfile loader ----
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    else:
        audio = audio[:, 0]

    if sr != sr_target:
        audio = torchaudio.transforms.Resample(sr, sr_target)(torch.from_numpy(audio)).numpy()

    # ---- processor input ----
    inputs = processor([audio], sampling_rate=sr_target, return_tensors="pt", padding=True)

    # ---- forward pass ----
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**batch)
        logits = outputs.logits            # <--- THIS is logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # ---- resolve label name ----
    id2label = model.config.id2label
    pred_id = int(np.argmax(probs))
    label_name = resolve_label_name(id2label, pred_id)

    # ---- produce probability mapping ----
    num_labels = logits.shape[-1]
    prob_dict = {
        resolve_label_name(id2label, i): float(probs[i])
        for i in range(num_labels)
    }

    return {
        "label": label_name,
        "probs": prob_dict
    }


def predict_folder(folder: str, ckpt_dir: str, out_json: str = "predictions.json", sr_target: int = 16000):
    results = []
    for p in sorted(glob.glob(os.path.join(folder, "*.wav"))):
        if not is_real_wav(p): continue
        r = predict_wav(p, ckpt_dir, sr_target=sr_target); r["path"] = p
        results.append(r)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved predictions to {out_json}")

# ============== CLI ==============
def main():
    parser = argparse.ArgumentParser(description="Wav2Vec2 SER (emotion from filename tag)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("inspect", help="Inspect emotion label distribution in flat folder")
    p_ins.add_argument("--source_dir", required=True)

    p_sp = sub.add_parser("split", help="Split flat folder into train/val/test by emotion")
    p_sp.add_argument("--source_dir", required=True)
    p_sp.add_argument("--target_dir", default="./dataset_split")
    p_sp.add_argument("--train_ratio", type=float, default=0.7)
    p_sp.add_argument("--val_ratio", type=float, default=0.15)
    p_sp.add_argument("--test_ratio", type=float, default=0.15)
    p_sp.add_argument("--seed", type=int, default=42)
    p_sp.add_argument("--no_csv", action="store_true")

    p_tr = sub.add_parser("train", help="Fine-tune Wav2Vec2 on split dataset")
    p_tr.add_argument("--data_root", default="./dataset_split")
    p_tr.add_argument("--model_name", default="facebook/wav2vec2-base")
    p_tr.add_argument("--out_dir", default="./ser_wav2vec2_ckpt")
    p_tr.add_argument("--seed", type=int, default=42)
    p_tr.add_argument("--sr_target", type=int, default=16000)
    p_tr.add_argument("--max_seconds", type=float, default=6.0)
    p_tr.add_argument("--batch_size", type=int, default=8)
    p_tr.add_argument("--epochs", type=int, default=10)
    p_tr.add_argument("--lr", type=float, default=3e-5)
    p_tr.add_argument("--weight_decay", type=float, default=0.01)
    p_tr.add_argument("--num_workers", type=int, default=0)
    p_tr.add_argument("--no_class_weights", action="store_true")

    p_te = sub.add_parser("test", help="Evaluate on test split")
    p_te.add_argument("--data_root", default="./dataset_split")
    p_te.add_argument("--ckpt_dir", default="./ser_wav2vec2_ckpt")
    p_te.add_argument("--sr_target", type=int, default=16000)
    p_te.add_argument("--max_seconds", type=float, default=6.0)
    p_te.add_argument("--batch_size", type=int, default=8)

    p_pw = sub.add_parser("predict_wav", help="Predict a single WAV")
    p_pw.add_argument("--wav_path", required=True)
    p_pw.add_argument("--ckpt_dir", default="./ser_wav2vec2_ckpt")
    p_pw.add_argument("--sr_target", type=int, default=16000)

    p_pf = sub.add_parser("predict_folder", help="Predict all WAVs in a folder")
    p_pf.add_argument("--folder", required=True)
    p_pf.add_argument("--ckpt_dir", default="./ser_wav2vec2_ckpt")
    p_pf.add_argument("--sr_target", type=int, default=16000)
    p_pf.add_argument("--out_json", default="predictions.json")

    args = parser.parse_args()

    if args.cmd == "inspect":
        inspect_source_dir(args.source_dir); return

    if args.cmd == "split":
        split_dataset(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            make_metadata_csv=(not args.no_csv),
        ); return

    if args.cmd == "train":
        train_ser(
            data_root=args.data_root,
            model_name=args.model_name,
            out_dir=args.out_dir,
            seed=args.seed,
            sr_target=args.sr_target,
            max_seconds=args.max_seconds,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            use_class_weights=(not args.no_class_weights),
        ); return

    if args.cmd == "test":
        evaluate_on_test(
            data_root=args.data_root,
            ckpt_dir=args.ckpt_dir,
            sr_target=args.sr_target,
            max_seconds=args.max_seconds,
            batch_size=args.batch_size,
        ); return

    if args.cmd == "predict_wav":
        out = predict_wav(args.wav_path, args.ckpt_dir, sr_target=args.sr_target)
        print(json.dumps(out, indent=2)); return

    if args.cmd == "predict_folder":
        predict_folder(args.folder, args.ckpt_dir, out_json=args.out_json, sr_target=args.sr_target); return

if __name__ == "__main__":
    main()
