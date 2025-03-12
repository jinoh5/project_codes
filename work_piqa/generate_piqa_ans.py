from datasets import load_dataset
import torch 
from transformers import GPT2LMHeadModel, GPT2LMHeadModelAugmented, GPT2Tokenizer, GPT2Config
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from bert_score import score
import argparse

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device, flush=True)

class PIQADataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=100, compare=True):
        """
        tokenizer: GPT-2 tokenizer.
        max_length: Maximum sequence length for truncation.
        """
        self.data = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length 
        self.compare = compare

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["goal"]
        sol1 = item["sol1"]
        sol2 = item["sol2"]
        label = item["label"]
        label_text = sol1 if label == 0 else sol2

        if self.compare == False: 
            input_text = f"Q: {prompt}\n A:"
            inputs = self.tokenizer(input_text,
                                    max_length=self.max_length, 
                                    truncation=True, 
                                    padding="max_length", 
                                    return_tensors="pt")
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0)
            }
        if self.compare == True: 
            return label_text

def strip_answer(answer):
    matching_txt = re.search(r'A: (.*)', answer)
    if matching_txt:
        return matching_txt.group(1).strip()
    else: 
        return matching_txt

def get_args():
    parser = argparse.ArgumentParser(description="Generate PIQA")
    parser.add_argument(
        "--model_name", type=str, default="untuned_GPT", help="Choose which model to upload"
    )  # options: tuned_GPT, tuned_memory
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    model_name = args.model_name
    # Load the dataset 
    dataset = load_dataset("piqa")

    # Prepare the test set: Since the test set doesn't have label, I am going to take about 20% of the data from the evaluation set to set up the test dataset
    validation = pd.DataFrame(dataset["validation"])
    test = validation[1470:]
    test = test.to_dict(orient='records')

    # Set up the tokenizer 
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set up Test 
    test_dataset = PIQADataset(test, tokenizer, compare=False)
    test_loader = DataLoader(test_dataset, batch_size=100)

    # Reference Answers 
    answer = PIQADataset(test, tokenizer, compare=True)
    answers = [answer[idx] for idx in range(len(answer))]

    if model_name == "tuned_GPT":    
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
        checkpoint = torch.load("/net/scratch/shared_memoryproject/gpt2_medium_checkpoint_3.pth")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Saved weights for GPT2-medium are uploaded")

    elif model_name == "untuned_GPT":
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)

    elif model_name == "tuned_memory":
        config = GPT2Config()
        config.n_embd = 1024
        config.n_layer = 16
        config.n_head = 16
        config.augmented_layer_idxs = [15]
        model = GPT2LMHeadModelAugmented.from_pretrained("/net/scratch/shared_memoryproject/memory_15_checkpoint_10.pth", config=config).to(device)
        checkpoint = torch.load("/net/scratch/shared_memoryproject/memory_15_checkpoint_10.pth")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Saved weights for Memory (15) - GPT2-medium")

    # Evaluation stage 
    model.eval()

    generated_outputs = []
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        output = model.generate(input_ids = input_ids,
                                attention_mask = attention_mask,
                                max_new_tokens=100, # you can adjust this 
                                num_return_sequences=1, # number of sequences to generate 
                                no_repeat_ngram_size=2, # prevent repeating n-grams 
                                top_k=50, # Sampling from top-k logits 
                                top_p=0.95, # nucleus sampling 
                                temperature=0.2, # control randomness (too much freedom could make bad decisions)
                                do_sample=True) # enable sampling 
        text = tokenizer.batch_decode(output, skip_special_tokens=True)
        generated_outputs.extend(text)

    # Saved outputs in text form 
    file_path = f'/home/yujinoh/project_codes/work_piqa/{model_name}_genoutput.txt'
    with open(file_path, 'w') as file:
        for item in generated_outputs:
            file.write(f"{item}\n")

    # Post-processing after generation before going into bert score 
    generated_answers = [strip_answer(text) for text in generated_outputs]
    generated_answers = [ans if ans is not None else "" for ans in generated_answers]
    # BERTScore 
    P, R, F1 = score(generated_answers, answers, lang="en", verbose=False)

    print(f"{model_name} Mean Precision: {torch.mean(P)}")
    print(f"{model_name} Mean Recall: {torch.mean(R)}")
    print(f"{model_name} Mean F1: {torch.mean(F1)}")

if __name__ == "__main__":
    main()