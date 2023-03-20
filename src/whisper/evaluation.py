from utils import load_split, prepare_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate
import torch


def map_to_result(batch):
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")

    with torch.no_grad():
        input_features = torch.tensor(batch["input_features"], device="cuda")
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        predicted_transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        batch["text"] = processor.batch_decode(batch["labels"], skip_special_tokens=True)
        batch["pred_str"] = predicted_transcript

    return batch

if __name__ == "__main__":
    
    # Load the model
    model_size = "small"
    model_checkpoint_path = "./whisper-small-zh/checkpoint-1000" 
    model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint_path)
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")

    # Load the test dataset
    test_data = load_split('data/generation/test.txt', 'data/generation/data')

    pres_test_data = test_data.map(prepare_dataset(processor=processor),
                                     remove_columns=test_data.column_names["test"], num_proc=8)
    
    res = pres_test_data.map(map_to_result, batched=True, batch_size=64)

    metric = evaluate.load("wer")

    wer = 100 * metric.compute(predictions=res["pred_str"], references=res["text"])
    print(f"test WER: {wer:.2f}")
