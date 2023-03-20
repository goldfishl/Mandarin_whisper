from transformers import Seq2SeqTrainingArguments
import datetime
import os
from tensorboardX import SummaryWriter


exp_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_size = "tiny"

path_config = {
    "output" : os.path.join("whisper", model_size, exp_name),
    "data_root" : os.path.join("data", "generation"),
    "split_file" : {},
}
path_config["eval"] = os.path.join(path_config["output"], "evaluation")
path_config["data_path"] = os.path.join(path_config["data_root"], "data")
path_config["split_file"]["train"] = os.path.join(path_config["data_root"], "train.txt")
path_config["split_file"]["val"] = os.path.join(path_config["data_root"], "val.txt")
path_config["split_file"]["test"] = os.path.join(path_config["data_root"], "test.txt")

os.makedirs(path_config["eval"], exist_ok=True)

# set training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=path_config["output"],
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=64,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=200,
    eval_steps=200,
    logging_steps=5,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)