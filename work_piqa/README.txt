First activate your conda environment, 
1. If you want to run untuned GPT-2 medium, then write in your command line: python run_untunedgpt2.py
2. If you want to run fine-tune GPT-2 medium, then write in your command line: python run_gpt2.py
3. If you want to run fine-tune GPT-2 meidum with memory layers, then write in your command line: python run_memorygpt2.py
    You can choose which layers you want to change from feed-forward layers to memory layers by typing:
        ex. --memory_layer 2 15