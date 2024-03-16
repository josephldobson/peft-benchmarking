from datasets import load_dataset, concatenate_datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import random

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

random.seed(SEED)

def preprocess_and_tokenize_data(tokenizer, examples, question_key, choices_key, answer_key):
    input_texts = []
    target_texts = []

    for question, choices, answer in zip(examples[question_key], examples[choices_key], examples[answer_key]):
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


def prepare_datasets(num_train, num_test):
    original_dataset = load_dataset(LOADED_DATASET, DATASET_CONFIG)
    merged_dataset = concatenate_datasets([original_dataset['test'], original_dataset['validation'], original_dataset['dev']])
    merged_dataset = merged_dataset.shuffle(seed=SEED)
    train_dataset = merged_dataset.select(range(num_train))
    test_dataset = merged_dataset.select(range(num_train, num_train + num_test))
    return train_dataset, test_dataset

def main():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.01,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_total_limit=SAVE_LIMIT,
        push_to_hub=False,
        # fp16=True, # Enable for FP16 training if supported by hardware
    )

    train_dataset, test_dataset = prepare_datasets(NUM_TRAIN_EXAMPLES, NUM_TEST_EXAMPLES)

    tokenized_train_dataset = train_dataset.map(lambda x: preprocess_and_tokenize_data(tokenizer, x, 'question', 'choices', 'answer'), batched=True)
    tokenized_test_dataset = test_dataset.map(lambda x: preprocess_and_tokenize_data(tokenizer, x, 'question', 'choices', 'answer'), batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
    )

    model.config.use_cache = False
    trainer.train()

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
