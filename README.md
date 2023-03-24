Fine-tuning the whisper model on the Mandarin dataset

# Results
Each model was trained for 1000 steps (about 5 epoch).

| Model Size | Parameters | Best Val WER | Test WER |
| --- | --- | --- | --- |
| tiny | 39M | 17.07 | 16.17 |
| base | 74M | 34.58 | 35.94 |
| small | 244M | 1.8 | 2.86 |

# Credits
The implementation is based on: https://huggingface.co/blog/fine-tune-whisper