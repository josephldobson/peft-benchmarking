from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from datasets import concatenate_datasets
from utilities import preprocess_function

MODEL_NAME = "google/flan-t5-small"
TASK_NAME = "mmlu_lora_testing"
OUTPUT_DIR = f"models/prompt-tuning-{TASK_NAME}"

# Preprocess and tokenize data
dataset = load_dataset('cais/mmlu','abstract_algebra')
merged_dataset = concatenate_datasets([dataset['test'], dataset['validation'], dataset['dev']])
merged_dataset = merged_dataset.shuffle(seed=42)
train_dataset = merged_dataset.select(range(100))
test_dataset = merged_dataset.select(range(100, 116))

processed_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
processed_test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

tokenizer = T5Tokenizer.from_pretrained("t5-small")

def tokenize_function(examples):
    inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=512)
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"],
            "labels": outputs["input_ids"]}

tokenized_train_dataset = processed_train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = processed_test_dataset.map(tokenize_function, batched=True)

model = T5ForConditionalGeneration.from_pretrained("t5-small")

lora_config = LoraConfig(
    r = 4,
    lora_alpha = 32,
    lora_dropout = 0.01,
    #bias = "none",
    task_type = TaskType.SEQ_2_SEQ_LM,
)

lora_model=get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=500,
    save_total_limit=3,
    # fp16=True, # uncomment this if GPU is available
    push_to_hub = False
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)

lora_model.config.use_cache = False
trainer.train()

peft_model_id=OUTPUT_DIR
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)