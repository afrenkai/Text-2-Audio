from datasets import load_dataset

if __name__ == "__main__":
    SAVE_DIR = './data/'
    import os
    os.makedirs(SAVE_DIR, exist_ok = True) #if dir not made make it else nothing
    hf_dataset = load_dataset('keithito/lj_speech')
    hf_dataset.save_to_disk(SAVE_DIR)