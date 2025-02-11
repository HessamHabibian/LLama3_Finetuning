# -*- coding: utf-8 -*-
!pip install transformers datasets evaluate peft trl bitsandbytes

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging
from peft import LoraConfig
from trl import SFTTrainer

base_model = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
guanaco_dataset = "mlabonne/guanaco-llama2-1k"
new_model = "llama-1.1B-chat-guanaco"

dataset = load_dataset(guanaco_dataset, split="train")
model = AutoModelForCausalLM.from_pretrained(base_model, device_map='auto')
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # pad sequences
tokenizer.padding_side = 'right'

# run inference
logging.set_verbosity(logging.CRITICAL)
prompt = "Who is Napoleon Bonaparte?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

peft_params = LoraConfig(lora_alpha=16, # multiplier of Lora output when its added to the full forward output
                         lora_dropout=0.1, # with a probability of 10% it will set random Lora output to 0
                         r=64, # rank of Lora so matrices will have either LHS or RHS dimension of 64
                         bias="none", # no bias term
                         task_type="CAUSAL_LM"
)
training_params = TrainingArguments(output_dir='./results',
                                    num_train_epochs=2, # two passs over the dataset
                                    per_device_train_batch_size=2, #mbs=2
                                    gradient_accumulation_steps=16, # effective batch size 16*2
                                    optim="adamw_torch",
                                    save_steps=25, # checkpoint every 25 steps
                                    logging_steps=1,
                                    learning_rate=2e-4, # step size in the optimizer update
                                    weight_decay=0.001,
                                    fp16=True, # 16 bit
                                    bf16=False, # not supported on V100
                                    max_grad_norm=0.3, #gradient clipping improves convergence
                                    max_steps=-1,
                                    warmup_ratio=0.03, # learning rate warmup
                                    group_by_length=True,
                                    lr_scheduler_type="cosine" # cosine lr scheduler
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params, # parameter efficient fine tuning AKA Lora
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False
)
import gc # garbage collection
gc.collect()
torch.cuda.empty_cache() # clean cache

trainer.train() # train the model
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

prompt = "Who is Napoleon Bonaparte?"
pipe = pipeline(task='text-generation', model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f'<s>[INST] {prompt} [/INST]')
print(result[0]['generated_text'])