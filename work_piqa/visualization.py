import numpy as np
import os
import matplotlib.pyplot as plt 

# WRITTEN BY JIN
def visualize(npy_savedir, eval1, eval2, eval3, eval4, loss1, loss2, loss3, loss4):

    epoch_range = np.arange(1,11,1)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    axs[0].plot(epoch_range, loss1, c='blue', label='fine-tuned gpt2-medium')
    axs[0].plot(epoch_range, loss2[:10], c='orange', label='gpt2+memory_layer: 15')
    axs[0].plot(epoch_range, loss3, c='g', label='gpt2+memory_layer: 2,15')
    axs[0].plot(epoch_range, loss4[:10], c='r', label='gpt2+memory_layer: 2,7,15')
    axs[0].set_ylim(bottom=0)
    axs[0].set_title("PIQA dataset: Training Loss")
    axs[0].set_ylabel("Training Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].legend()

    axs[1].plot(epoch_range, eval1, c='blue',label='fine-tuned gpt2-medium')
    axs[1].plot(epoch_range, eval2[:10], c='orange',label='gpt2+memory_layer: 15')
    axs[1].plot(epoch_range, eval3, c='g', label='gpt2+memory_layer: 2,15')
    axs[1].plot(epoch_range, eval4[:10], c='r', label='gpt2+memory_layer: 2,7,15')
    axs[1].axhline(0.5, linestyle='--', c='black')
    axs[1].set_ylim(0,1)
    axs[1].set_title("PIQA dataset: Evaluation Accuracy")
    axs[1].set_ylabel("Evaluation Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].legend()

    plot_filename = "total_plot.png"
    plt.savefig(os.path.join(npy_savedir, plot_filename))

def main():
    saved_dir = "/home/yujinoh/project_codes/work_piqa/outputs"
    new_gpt2_medium_eval_acc = np.load(os.path.join(saved_dir, "new_gpt2_medium_eval_acc.npy"))
    print("new_gpt2_medium_eval_acc", new_gpt2_medium_eval_acc)
    memory_2715_eval_acc = np.load(os.path.join(saved_dir, "memory_2715_eval_acc.npy"))
    print("memory_2715_eval_acc", memory_2715_eval_acc)
    memory_215_eval_acc = np.load(os.path.join(saved_dir, "memory_2_15_eval_acc.npy"))
    print("memory_215_eval_acc", memory_215_eval_acc)
    memory_15_eval_acc = np.load(os.path.join(saved_dir, "memory_15_eval_acc.npy"))
    print("memory_15_eval_acc", memory_15_eval_acc)

    new_gpt2_medium_training_loss = np.load(os.path.join(saved_dir, "new_gpt2_medium_training_loss.npy"))
    memory_2715_training_loss = np.load(os.path.join(saved_dir, "memory_2715_training_loss.npy"))
    memory_215_training_loss = np.load(os.path.join(saved_dir, "memory_2_15_training_loss.npy"))
    memory_15_training_loss = np.load(os.path.join(saved_dir, "memory_15_training_loss.npy"))

    visualize(saved_dir, 
              new_gpt2_medium_eval_acc, memory_15_eval_acc, 
              memory_215_eval_acc, memory_2715_eval_acc, 
              new_gpt2_medium_training_loss, memory_15_training_loss, 
              memory_215_training_loss, memory_2715_training_loss)


if __name__ == "__main__":
    main()
