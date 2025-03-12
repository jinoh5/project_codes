from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModelAugmented, GPT2Config
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch

import random

def strip_strip(x):
    return x.strip()

def read_nb():
    with open("train_nb.txt") as f:
        nb = list(map(int, map(strip_strip, f.readlines())))
    return nb

def read_sent():
    readings = []
    with open("train_sent.txt") as f:
        for article_len in nb:
            readings.append("")
            for _ in range(article_len):
                readings[-1] += f.readline()
            readings[-1] = readings[-1].replace("\n", "")
    return readings


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nb = read_nb()
    sent = read_sent()

    random.shuffle(sent)
    sent = sent[:500000]

    # Initialize tokenizer and add pad token
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    print("Loaded Data", flush=True)

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),  # Removed .to(device)
                'attention_mask': encoding['attention_mask'].squeeze(0)  # Removed .to(device)
            }

    dataset = TextDataset(sent, tokenizer, 512)

    print("Dataset Built", flush=True)

    batch_size = 8

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Dataloader Built", flush=True)

    config = GPT2Config()

    config.n_embd=1024
    config.n_layer=16
    config.n_head=16

    config.augmented_layer_idxs = [2,15]

    # Initialize model and resize embeddings
    model = GPT2LMHeadModelAugmented.from_pretrained("gpt2-medium", config=config)
    model.resize_token_embeddings(len(tokenizer))  # Resize for new tokens

    output_dir = "gen_model_mem/"

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        save_strategy="steps",         # Set save strategy to "steps"
        save_steps=5000,               # Save checkpoint every 5000 steps (increased from 500)
        save_total_limit=1,            # Keep only the 2 most recent checkpoints
        logging_dir="./logs",            # Directory for storing logs
        logging_steps=500,
    )

    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
    )

    print("Starting Training", flush=True)

    trainer.train()
    model.save_pretrained(output_dir, safe_serialization=False) # ADD this line instead
    tokenizer.save_pretrained(output_dir)  # Save the tokenizer

    print("Training Done", flush=True)

    def load_model(model_path):
        model = GPT2LMHeadModelAugmented.from_pretrained(model_path)
        return model

    def load_tokenizer(tokenizer_path):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Load from saved directory
        return tokenizer

    def generate_text(sequence, max_length):
        model_path = output_dir
        model = load_model(model_path).to(device)
        tokenizer = load_tokenizer(model_path)
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        ids = tokenizer.encode(f'{sequence}', return_tensors='pt').to(device)
        final_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,  # Now correctly set
            top_k=50,
            top_p=0.95,
        )
        print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))

    sequence = "Michael Jordan"
    max_len = 40
    generate_text(sequence, max_len)