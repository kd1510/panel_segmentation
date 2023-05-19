from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from prepare_data import generate_train_val_ds
from metrics import compute_metrics
from transforms import train_transforms, val_transforms

checkpoint = "nvidia/mit-b0"
model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint)

train_dataset, val_dataset = generate_train_val_ds()
train_dataset = train_dataset.set_transform(train_transforms)
val_dataset = val_dataset.set_transform(val_transforms)

training_args = TrainingArguments(
    output_dir="segformer-b0-solar-panels",
    learning_rate=6e-5,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    remove_unused_columns=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()