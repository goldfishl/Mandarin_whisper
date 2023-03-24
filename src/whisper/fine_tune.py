from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Seq2SeqTrainer
from datasets import DatasetDict
from utils import load_split, prepare_dataset, compute_metrics
from utils import map_to_result
from config import training_args, model_size, path_config
import torch
import os
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch



if __name__ == "__main__":

    # Load the pre-trained model
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}")

    # Load the dataset
    generation = DatasetDict()
    generation["train"] = load_split(path_config["split_file"]["train"], path_config["data_path"])
    generation["val"] = load_split(path_config["split_file"]["val"], path_config["data_path"])
    generation["test"] = load_split(path_config["split_file"]["test"], path_config["data_path"])
    
    # # debug
    # from datasets import Dataset, Audio
    # data_path = os.path.join("data", "generation", "data")
    # generation["train"] = Dataset.from_dict({'audio': [os.path.join(data_path, "0.wav")], 'sentence': ["后顶,申脉,阿是穴,后溪,天柱"]}).cast_column("audio", Audio())
    # generation["val"] = Dataset.from_dict({'audio': [os.path.join(data_path, "2577.wav"), os.path.join(data_path, "77.wav")], 'sentence': ["四神聪,印堂,悬钟,神庭,太溪,百会", "百会,四神聪,太冲,内关,阿是穴"]}).cast_column("audio", Audio())
    # generation["test"] = Dataset.from_dict({'audio': [os.path.join(data_path, "325.wav"), os.path.join(data_path, "8840.wav")], 'sentence': ["肩髎,曲池,肩贞,合谷,阳陵泉,肩髃,阿是穴", "内关,日月,足三里,阳陵泉,胆俞"]}).cast_column("audio", Audio())

    # Load the processor
    # processor is a wrapper around the feature extractor and the tokenizer
    # since whisper model is a encoder-decoder model, we don't need to train a tokenizer from scratch
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")

    # Preprocess the dataset
    generation = generation.map(prepare_dataset(processor=processor), 
                                remove_columns=generation.column_names["train"], num_proc=16)

    # Create the data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # set forced_bos_token_id and suppress_tokens
    # forced_bos_token_id is used to specify the start token of the decoder, which is the language and task prompt
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="Chinese", task="transcribe")
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="Chinese", task="transcribe")
    model.config.suppress_tokens = []
    model.config.use_cache = False



    # initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=generation["train"],
        eval_dataset=generation["val"],
        data_collator=data_collator,
        compute_metrics=compute_metrics(processor.tokenizer),
        tokenizer=processor.feature_extractor,
    )

    # train the model
    trainer.train()

    # evaluate the model
    res = generation["test"].map(map_to_result(model, processor), batched=True, batch_size=64)

    metric = evaluate.load("wer")

    wer = 100 * metric.compute(predictions=res["pred_str"], references=res["text"])
    print(f"Test WER: {wer:.2f}")

    with open(os.path.join(path_config["output"], "test_wer.txt"), "w") as f:
        f.write(f"Test WER: {wer:.2f}")
