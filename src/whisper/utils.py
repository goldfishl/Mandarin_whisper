import os
from datasets import Dataset, Audio
import datetime
import evaluate
from config import path_config
import torch


def load_split(split_file, data_path):
    with open(split_file, 'r') as f:
        wav_files = []
        labels = []
        for line in f:
            wav_files.append(os.path.join(data_path, line.strip()))
            label_file, _ = os.path.splitext(line)
            with open(os.path.join(data_path, label_file + '.txt'), 'r') as f:
                transcript = f.read().strip()
                # change tab split to vertical bar split
                transcript = transcript.replace('\t', ',')
                labels.append(transcript)


    return Dataset.from_dict({'audio': wav_files, 'sentence': labels}).cast_column("audio", Audio())


def prepare_dataset(processor):
    def _prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch
    return _prepare_dataset


def compute_metrics(tokenizer):
    metric = evaluate.load("wer")
    def _compute_metrics(pred):
        time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        save_list_to_txt(os.path.join(path_config['eval'], f'{time}_val_pred.txt'), pred_str)
        save_list_to_txt(os.path.join(path_config['eval'], f'{time}_val_ref.txt'), label_str)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    return _compute_metrics

def map_to_result(model, processor):
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")
    
    def _map_to_result(batch):
        with torch.no_grad():
            input_features = torch.tensor(batch["input_features"], device="cuda")
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            predicted_transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)

            batch["text"] = processor.batch_decode(batch["labels"], skip_special_tokens=True)
            batch["pred_str"] = predicted_transcript
            
            save_list_to_txt(os.path.join(path_config['eval'], f'test_pred.txt'), predicted_transcript)
            save_list_to_txt(os.path.join(path_config['eval'], f'test_ref.txt'), batch["text"])

        return batch

    return _map_to_result

def save_list_to_txt(filename, data_list):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in data_list:
            file.write(f"{item}\n")