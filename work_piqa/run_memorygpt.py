from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2LMHeadModelAugmented, GPT2Tokenizer, GPT2Config
from torch.nn import CrossEntropyLoss
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os 
import pandas as pd
from datetime import datetime

# WRITTEN BY JIN 
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device, flush=True)

class PIQADataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128, train=True):
        """
        tokenizer: GPT-2 tokenizer.
        max_length: Maximum sequence length for truncation.
        """
        self.data = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length 
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Tokenizes and returns input tensors for GPT-2."""
        item = self.data[idx]
        label = item["label"]
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

def save_checkpoint(model, optimizer, epoch, loss, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    checkpoint = { 
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Check point saved to {filename}", flush=True)

def visualize(layer_name, npy_savedir, training_loss, eval_acc):
    epoch_range = np.arange(1,len(training_loss)+1,1)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    axs[0].plot(epoch_range, training_loss, c='r')
    axs[0].set_ylim(bottom=0)
    axs[0].set_title("GPT2 + Memory (random init) PIQA: Training Loss")
    axs[0].set_ylabel("Training Loss")
    axs[0].set_xlabel("Epoch")

    axs[1].plot(epoch_range, eval_acc, c='b')
    axs[1].set_ylim(0,1)
    axs[1].set_title("GPT + Memory (random init) PIQA: Eval Accuracy")
    axs[1].set_ylabel("Evaluation Accuracy")
    axs[1].set_xlabel("Epoch")

    plot_filename = f"memory_{layer_name}_training_and_eval_plot.png" # CHANGE FILENAME 
    plt.savefig(os.path.join(npy_savedir, plot_filename))

def get_args():
    parser = argparse.ArgumentParser(description="GPT2_w_memory")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="How many number of epochs?"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=5e-5, help="What is the learning rate?"
    )
    parser.add_argument(
        "-p", "--patience", type=int, default=3, help="What is the patience level?"
    )
    parser.add_argument(
        "--memory_layer", type=int, nargs="+", help="Which layers you want to change to memory layers?"
    ) # Write down
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    num_epochs = args.num_epochs
    patience = args.patience
    learning_rate = args.learning_rate
    memory_layer = args.memory_layer 

    config = GPT2Config()
    config.n_embd = 1024
    config.n_layer = 16
    config.n_head = 16
    config.augmented_layer_idxs = memory_layer 

    layer_name = ''
    for i in range(len(memory_layer)):
        layer_name += str(memory_layer[i])
    
    # Tokenize the dataset 
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("piqa")
    print("Dataset loaded", flush=True)
    validation = pd.DataFrame(dataset["validation"])
    val_data = validation[:1470].to_dict(orient='records')

    train_dataset = PIQADataset(dataset["train"], tokenizer, train=True)
    val_dataset = PIQADataset(val_data, tokenizer, train=False)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    # Model Instatiation 
    model = GPT2LMHeadModelAugmented.from_pretrained("gpt2-medium", config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    save_dir = '/net/scratch/yujinoh/checkpoints'
    npy_savedir = '/home/yujinoh/project_codes/work_piqa'
    # counter = 0 
    # best_acc = 0 

    # Separate storage for training loss 
    memory_training_loss = []
    memory_eval_acc = []

    model.train()
    start_time = datetime.now()
    print("Start Time", start_time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    for epoch in range(num_epochs):
        total_loss = 0 
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels=input_ids) # this labels will shift it automatically
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        save_checkpoint(model, optimizer, epoch, loss, save_dir, f"memory_{layer_name}_checkpoint_{epoch+1}.pth") 
        train_loss = total_loss/len(train_loader)
        print(f"Epoch num: {epoch+1}, Training Loss: {train_loss}", flush=True)

        acc = evaluate(model, val_loader, tokenizer)
        memory_training_loss.append(train_loss)
        memory_eval_acc.append(acc)

        # if acc > best_acc:
        #     best_acc = acc 
        #     counter = 0 
        # else:
        #     counter += 1 
        #     if counter >= patience: 
        #         print(f"Early stopping at epoch {epoch+1}", flush=True)

        #         break 
    
    # Visualization 
    end_time = datetime.now()
    print("End Time", end_time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    np.save(os.path.join(npy_savedir,f"memory_{layer_name}_training_loss"), memory_training_loss) 
    np.save(os.path.join(npy_savedir, f"memory_{layer_name}_eval_acc"), memory_eval_acc)
    visualize(layer_name, npy_savedir, memory_training_loss, memory_eval_acc)
    print("Evaluation ACC:", memory_eval_acc, flush=True)

if __name__ == "__main__":
    main()