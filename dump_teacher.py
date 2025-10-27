import torch, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

OUT = "teacher_sst2_logits.pt"
MODEL_ID = "textattack/bert-base-uncased-SST-2"  # already fine-tuned on SST-2

def make_loader(split, tok, max_len=128, bs=64):
    ds = load_dataset("glue", "sst2")[split]
    def _tok(batch):
        return tok(batch["sentence"], truncation=True, padding="max_length", max_length=max_len)
    ds = ds.map(_tok, batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
    return ds, loader

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
    model.eval()

    out = {}
    for split in ["train","validation"]:
        ds, loader = make_loader(split, tok)
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Teacher {split}"):
                batch = {k:v.to(device) for k,v in batch.items() if k!="label"}
                logits = model(**batch).logits
                all_logits.append(logits.cpu())
                all_labels.append(loader.dataset["label"][len(all_labels)*loader.batch_size:
                                                         len(all_labels)*loader.batch_size+logits.size(0)])
        out[split] = {
            "logits": torch.cat(all_logits, dim=0),
            "labels": torch.tensor(list(sum(all_labels, [])), dtype=torch.long)
        }

    torch.save(out, OUT)
    print(f"Saved: {OUT}")

if __name__ == "__main__":
    main()
