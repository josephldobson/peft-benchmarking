from datasets import load_dataset, concatenate_datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import random
import pandas as pd

MODEL_NAME = "t5-small"
LOADED_DATASET = "cais/mmlu"
DATASET_CONFIG = "abstract_algebra"
OUTPUT_DIR = "models/mmlu_lora_testing"
MAX_LENGTH = 512
NUM_TRAIN_EXAMPLES = 100
NUM_TEST_EXAMPLES = 16
SEED = 42
TRAIN_EPOCHS = 100
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
LOGGING_STEPS = 10
SAVE_LIMIT = 3

dataset = load_dataset("sordonia/flan-10k-flat")
flan_dict = pd.read_csv("flan_collection_info.csv")

multi_choice_qa_tasks_list = flan_dict.loc[flan_dict["Generic Task Category"] == "Multiple-Choice QA (no trivia knowledge required)"]["Specific Task Category"].drop_duplicates().tolist()
multi_choice_qa_tasks_set = set(multi_choice_qa_tasks_list)
mc_qa_dataset = dataset.filter(lambda r: r["task_name"] in multi_choice_qa_tasks_set, num_proc=10)
mc_qa_trainset = mc_qa_dataset.filter(lambda r: r["split"] == "train")
mc_qa_testset = mc_qa_dataset.filter(lambda r: r["split"] == "test")

print(mc_qa_trainset)
print(mc_qa_testset)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def tokenize_function(tokenizer, x):
    input_texts = []
    target_texts = []

    for question, choices, answer in zip(x["question"], x["choices"], x["answer"]):
        input_text = f"question: {question} options: {', '.join(choices)}"
        target_text = choices[answer]
        input_texts.append(input_text)
        target_texts.append(target_text)

    tokenized_inputs = tokenizer(input_texts, padding="max_length", truncation=True, max_length=MAX_LENGTH)
    tokenized_targets = tokenizer(target_texts, padding="max_length", truncation=True, max_length=MAX_LENGTH)

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_targets["input_ids"],
    }

tokenized_trainsets = mc_qa_trainset.map(
    lambda x: tokenize_function(tokenizer, x),
    batched=True
)

tokenized_testsets = mc_qa_testset.map(
    lambda x: tokenize_function(tokenizer, x),
    batched=True
)

lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.01,
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-3,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_trainsets,
    # eval_dataset=tokenized_testsets,
    tokenizer=tokenizer,
)

model.config.use_cache = False
trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)
