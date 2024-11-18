#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=50g
#SBATCH -J "TTS Transformer"
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sppradhan/Text-2-Audio/transformer_tts_%j.txt

module load cuda
module load python/3.10.13
source ~/Text-2-Audio/kdd4/bin/activate
python main.py