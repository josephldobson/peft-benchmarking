from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import PrefixTuningConfig, LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType


MODEL_NAME = "google/flan-t5-small"
TASK_NAME = "task599_cuad_question_generation"
OUTPUT_DIR = f"prompt-tuning-{TASK_NAME}"


# use the validation dataset for now because it's a manageable size
dataset = load_dataset("sordonia/flan-10k-flat", split="train") 
qa_dataset = dataset.filter(lambda rec: rec["task_name"] == TASK_NAME)
print(qa_dataset)

# prepare model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, 
                                            #   load_in_8bit=True, 
                                              device_map="auto")

# quanitization for memory efficiency
# model = prepare_model_for_int8_training(model)

# add lora layer to train
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

prefix_tuning_config = PrefixTuningConfig(
    peft_type="PREFIX_TUNING",
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_transformer_submodules=1,
    num_attention_heads=12,
    num_layers=12,
    encoder_hidden_size=768,
)

# add peft component to model
model = get_peft_model(model, prefix_tuning_config)
model.print_trainable_parameters()

# # padding for inputs
# # we want to ignore tokenizer pad token in the loss
# label_pad_token_id = -100
# # Data collator
# data_collator = DataCollatorForSeq2Seq(
#     tokenizer,
#     model=model,
#     label_pad_token_id=label_pad_token_id,
#     pad_to_multiple_of=8
# )

# prepare training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
	auto_find_batch_size=True,
    learning_rate=1e-3,
    max_steps=5,
    num_train_epochs=1,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_strategy="steps",
    logging_steps=5,
    save_strategy="no",
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=qa_dataset,
    # eval_dataset=eval_dataset,
)

model.config.use_cache = False

trainer.train()

# Save model & tokenizer results
peft_model_id=OUTPUT_DIR
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
