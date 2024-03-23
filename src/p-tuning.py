from prefix_tuning import prepare_datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict, load_from_disk
from peft import PromptEncoder, PromptEncoderConfig, get_peft_model
import pandas as pd
import string

OUTPUT_DIR = "models/flan_10k_t5small_p-tuning"
MODEL_NAME = "t5-small"

dataset = load_dataset("sordonia/flan-10k-flat", split='train[:5%]')
flan_dict = pd.read_csv("flan_collection_info.csv")

multi_choice_qa_tasks_list = flan_dict.loc[flan_dict["Generic Task Category"] == "Multiple-Choice QA (no trivia knowledge required)"]["Specific Task Category"].drop_duplicates().tolist()
multi_choice_qa_tasks_set = set(multi_choice_qa_tasks_list)
mc_qa_dataset = dataset.filter(lambda r: r["task_name"] in multi_choice_qa_tasks_set)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    model_inputs = tokenizer(text=examples["source"], text_target=examples["target"], truncation=True, max_length=tokenizer.model_max_length)
    return model_inputs

tokenized_datasets = mc_qa_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["source", "task_name", "task_source", "template_type", "template_idx", "split", "target"],
)
print(tokenized_datasets)
config = PromptEncoderConfig(
    peft_type="P_TUNING",
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    encoder_hidden_size=128,
)

model = get_peft_model(model, config)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

model.config.use_cache = False
trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
