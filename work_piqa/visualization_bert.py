import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd

# WRITTEN BY JIN 
def main():
    data = [0.84, 0.87, 0.86]
    labels = ['GPT-2 Medium\n(Untuned)', 'GPT-2 Medium\n(Finetuned)', 'GPT-2 Medium +\n Memory [2,15] (Fine-tuned)']
    plt.figure(figsize=(8,6))
    # sns.set_theme(style="whitegrid")
    plt.bar(labels, data, color=['blue', 'green', 'orange']) 
    for index, value in enumerate(data):
        plt.text(index, value + 0.005, f"{value:.2f}", ha='center', fontsize=12)
    plt.grid(True, axis='y', color='lightgray', linestyle='-', linewidth=0.5)
    plt.ylim(0.6, 0.9) 
    plt.ylabel("BERTScore (F1)", fontsize=12)
    plt.title("BERTScore on PIQA Testing Set", fontsize=18, pad=10)
    plt.xticks(fontsize=12)
    plt.savefig("/home/yujinoh/project_codes/work_piqa/bertscore_piqa.jpeg")

if __name__ == "__main__":
    main()
