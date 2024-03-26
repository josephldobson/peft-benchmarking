from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict, load_from_disk
from peft import PromptEncoder, PromptEncoderConfig, get_peft_model
import pandas as pd


def task():
    NUM_PROCS = os.cpu_count() - 1
    PEFT_METHOD = "P_TUNING"
    OUTPUT_DIR = "models/t5small_p-tuning"
    MODEL_NAME = "t5-small"
    BATCH_SIZE = 32
    NUM_EPOCHS = 4

    dataset = load_dataset("sordonia/flan-10k-flat", split="train[:1%]+train[-1%:]")
    flan_dict = pd.read_csv("flan_collection_info.csv")

    multi_choice_qa_tasks_list = flan_dict.loc[flan_dict["Generic Task Category"] == "Multiple-Choice QA (no trivia knowledge required)"]["Specific Task Category"].drop_duplicates().tolist()
    multi_choice_qa_tasks_set = set(multi_choice_qa_tasks_list)
    mc_qa_dataset = dataset.filter(lambda r: r["task_name"] in multi_choice_qa_tasks_set, num_proc=NUM_PROCS)
    mc_qa_trainset = mc_qa_dataset.filter(lambda r: r["split"] == "train")
    mc_qa_testset = mc_qa_dataset.filter(lambda r: r["split"] == "test")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    def tokenize_function(tokenizer, x):
        tokenized_inputs = tokenizer(x['source'], padding="max_length", truncation=True, max_length=model.config.max_length)
        tokenized_targets = tokenizer(x['target'], padding="max_length", truncation=True, max_length=model.config.max_length)
        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_targets["input_ids"]
        }

    tokenized_trainsets = mc_qa_trainset.map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True
    )

    tokenized_testsets = mc_qa_testset.map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True
    )

    config = PromptEncoderConfig(
        peft_type="P_TUNING",
        task_type="SEQ_2_SEQ_LM",
        # num_virtual_tokens=20,
        # encoder_hidden_size=128
    )

    model = get_peft_model(model, config)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_trainsets,
        eval_dataset=tokenized_testsets,
        tokenizer=tokenizer,
    )

    model.config.use_cache = False
    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)

if __name__ == '__main__':
    task()
