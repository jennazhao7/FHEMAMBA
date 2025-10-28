import torch, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

OUT = "teacher_sst2_logits.pt"
MODEL_ID = "textattack/bert-base-uncased-SST-2"  # already fine-tuned on SST-2

def make_loader(split, tok, max_len=128, bs=128):
    ds = load_dataset("glue", "sst2")[split]
    def _tok(batch):
        return tok(batch["sentence"], truncation=True, padding="max_length", max_length=max_len)
    ds = ds.map(_tok, batched=True, num_proc=4)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    return ds, loader

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print(f"Model loaded: {MODEL_ID}")

    out = {}
    for split in ["train","validation"]:
        print(f"\nProcessing {split}...")
        ds, loader = make_loader(split, tok, max_len=128, bs=128)
        print(f"Dataset size: {len(ds)}, batches: {len(loader)}")
        
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Teacher {split}")):
                labels = batch["label"].cpu()
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                all_logits.append(logits.cpu())
                all_labels.append(labels)
        out[split] = {
            "logits": torch.cat(all_logits, dim=0),
            "labels": torch.cat(all_labels, dim=0)
        }
        print(f"✓ {split}: logits={out[split]['logits'].shape}, labels={out[split]['labels'].shape}")

    torch.save(out, OUT)
    print(f"\n✓ Saved: {OUT}")

if __name__ == "__main__":
    main()
