from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2LMHeadModelAugmented, GPT2Config
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from bert_score import score

import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def strip_strip(x):
    return x.strip()

def read_nb():
    with open("test_nb.txt", encoding="latin-1") as f:
        nb = list(map(int, map(strip_strip, f.readlines()))) 
    return nb

def read_sent():
    readings = []
    with open("test_sent.txt", encoding="latin-1") as f:
        for idx, article_len in enumerate(nb):
            readings.append("")
            for line in range(article_len):
                readings[-1] += f.readline()            
            readings[-1] = readings[-1].replace("\n", "")            
    return readings

def main():

    config = GPT2Config.from_pretrained("gen_model_mem/")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    tokenizer.pad_token = '<|pad|>'

    config.pad_token_id = tokenizer.pad_token_id
    config.vocab_size = len(tokenizer) 

    model = GPT2LMHeadModelAugmented.from_pretrained(
        "gen_model_mem/",
        config=config,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id

    def generate_text(sequence, max_length):
        ids = tokenizer.encode(f'{sequence}', return_tensors='pt').to(DEVICE)
        final_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            top_k=50,
            top_p=0.95,
        )
        return tokenizer.decode(final_outputs[0], skip_special_tokens=True)


    def find_n_space(text):
        space_count = 0
        index = -1
        for i, char in enumerate(text):
            if char == ' ':
                space_count += 1
                if space_count == 4:
                    index = i
                    break
        return index

    sent_sample = sent

    drops = []

    f = open("memory_zipper/memory_output.txt", "w")

    f1s = []
    generateds = []
    for i, sample in enumerate(sent_sample):
        idx = find_n_space(sample)
        if idx == -1:
            drops.append(i)
            print("FACK!", sample)
            continue
        
        start_seq = sample[:find_n_space(sample)]

        max_len = 100
        
        generated_text = generate_text(start_seq, max_len)
        f.write(generated_text + "\n")
        f.write("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

        sample = sample[:len(generated_text)]
        f.write(sample + "\n")
        f.write("-------------------------------------------------------------------\n")
    
        _, _, F1 = score([generated_text], [sample], lang="en")
        f1s.append(F1[0].item())

        if i % 100 == 0:
            print("Progress:", i / len(sent_sample), flush = True)

    f.close()
    

    # print(f"F1: {F1.tolist()}")

    with open("memory_zipper/bert_score.txt", "w") as f:
        f.write(str(sum(f1s) / len(f1s)) + "\n")

    print(sum(f1s) / len(f1s))




if __name__ == "__main__":

    random.seed(11)

    nb = read_nb()

    sent = read_sent()

    sent = [s for s in sent if s]

    random.shuffle(sent)

    main()
