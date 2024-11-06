#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=50g
#SBATCH -J "TTS Simple Test"
#SBATCH -p short
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sppradhan/TEXT-2-AUDIO/tts_simple_%j.txt

module load cuda
module load python/3.10.13
source ~/TEXT-2-AUDIO/kdd4/bin/activate
cd Models
python main.py