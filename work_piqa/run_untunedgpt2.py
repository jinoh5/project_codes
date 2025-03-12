from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.nn import CrossEntropyLoss
import torch
import torch.optim as optim
import numpy as np
import json
import argparse
import os 

# WRITTEN BY JIN 
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device, flush=True)

class PIQADataset(Dataset):
    def __init__(self, jsonl_file, lst_file, tokenizer, max_length=128, train=True):
        """
        data_path: Path to the PIQA dataset JSON file.
        tokenizer: GPT-2 tokenizer.
        max_length: Maximum sequence length for truncation.
        """
        self.jsonl_file = jsonl_file 
        self.lst_file = lst_file 
        self.tokenizer = tokenizer
        self.max_length = max_length 
        self.train = train

        with open(jsonl_file, "r") as f:
            self.data = [json.loads(line.strip()) for line in f.readlines()]
        with open(lst_file, "r") as f:
            self.labels = [int(line.strip()) for line in f.readlines()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Tokenizes and returns input tensors for GPT-2."""
        item = self.data[idx]
        label = self.labels[idx]

        prompt = item["goal"]
        sol1 = item["sol1"]
        sol2 = item["sol2"]
        label_text = sol1 if label == 0 else sol2

        if self.train == True:
            input_text = f"Q: {prompt}\n A: {label_text}"
            inputs = self.tokenizer(input_text,
                                    max_length=self.max_length, 
                                    truncation=True, 
                                    padding="max_length", 
                                    return_tensors="pt")
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0)
            }
        else:
            prompt_ans1 = f"Q: {prompt}\n A: {sol1}"
            prompt_ans2 = f"Q: {prompt}\n A: {sol2}"

            enc1 = self.tokenizer(prompt_ans1,
                                  max_length = self.max_length,
                                  truncation = True,
                                  padding= "max_length",
                                  return_tensors="pt")
            enc2 = self.tokenizer(prompt_ans2, 
                                  max_length = self.max_length,
                                  truncation = True,
                                  padding = "max_length",
                                  return_tensors="pt")
            return {
                "input_ids1": enc1["input_ids"].squeeze(0),
                "attention_mask1": enc1["attention_mask"].squeeze(0),
                "input_ids2": enc2["input_ids"].squeeze(0),
                "attention_mask2": enc2["attention_mask"].squeeze(0),
                "label": label
            }

def compute_log_probs(model, tokenizer, input_ids, attention_mask):
    loss_fn = CrossEntropyLoss(reduction = 'sum', ignore_index = tokenizer.pad_token_id)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask = attention_mask)
        # outputs.logits = (batch size, seq len, vocab size) => why is there (batch size, x? , sequence_length, vocab size) 
        # input_ids shape should be (batch_size, sequence_length)
        logits = outputs.logits
        log_probs = -loss_fn(logits[:, :-1, :].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))
    return log_probs.item()

def evaluate(model, dataloader, tokenizer):
    model.eval()
    correct = 0
    total = 0
    new_label = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids1, input_ids2 = batch["input_ids1"].to(device), batch["input_ids2"].to(device)
            attention_mask1, attention_mask2 = batch["attention_mask1"].to(device), batch["attention_mask2"].to(device)
            labels = batch["label"].to(device) # 100, 128

            log_prob1 = compute_log_probs(model, tokenizer, input_ids1, attention_mask1)
            log_prob2 = compute_log_probs(model, tokenizer, input_ids2, attention_mask2)
            
            pred = 0 if log_prob1 > log_prob2 else 1
            labels = labels.cpu().numpy()
            new_label.append(labels)
            correct += (pred == labels).sum().item()
            total += 1
            
    acc = correct/total 
    print("Evaluation Accuracy", acc, flush=True)
    return acc


def main():
    # Tokenize the dataset 
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    # train_dataset = PIQADataset("train.jsonl", "train-labels.lst", tokenizer, train=True)
    val_dataset = PIQADataset("valid.jsonl", "valid-labels.lst", tokenizer, train=False)

    # train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    # Model Instatiation 
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    acc = evaluate(model, val_loader, tokenizer)
    print("acc", acc, flush=True)
    
if __name__ == "__main__":
    main()