# #ТЬЮНИНГ С САФФИКСОМ И ИНСТРУКЦИЕЙ
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch

file_path = "Instructions_new_new.xlsx"
df = pd.read_excel(file_path)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_data = Dataset.from_pandas(train_df)
valid_data = Dataset.from_pandas(test_df)


model_id = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to('cpu')


def preprocess(data, tokenizer):
    prompts = data['Input']
    responses = data['Output']
    instruction = "Исправь ошибку и объясни, что было неправильно:"
    suffix = "\n### Исправленный текст:\n"

    combined = []
    for prompt, response in zip(prompts, responses):
        combined_text = f"### Инструкция:\n{instruction}\n\n### Входные данные:\n{prompt}\n\n### Ответ:\n{response}{suffix}"
        combined.append(combined_text)

    tokenized = tokenizer(combined, truncation=True, padding='max_length', max_length=256, return_tensors="pt")
    labels = tokenized['input_ids'].clone()
    labels[tokenized['attention_mask'] == 0] = -100

    return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask'], 'labels': labels}

# Применение предварительной обработки к датасету
dataset = DatasetDict({
    'train': train_data,
    'validation': valid_data,
})

dataset = dataset.map(
    lambda d: preprocess(d, tokenizer),
    batched=True
)

# параметры
training_args = TrainingArguments(
    output_dir='./mistakes_det',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    logging_dir='./logs',
    num_train_epochs=4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    logging_steps=100,
    eval_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
)

# Определение Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
)


trainer.train()
# trainer.train(resume_from_checkpoint='./saved_next/checkpoint-486')
trainer.save_model('./mistakes_det')
tokenizer.save_pretrained('./mistakes_det')




#тест модели С ЛЕЙБЛАМИ

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = './labels/checkpoint-1944'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to('cpu')


def preprocess_for_inference(prompt, instruction, suffix, tokenizer, max_length):
    formatted_text = f"### Инструкция:\n{instruction}\n\n### Входные данные:\n{prompt}\n\n### Ответ:\n{suffix}"
    inputs = tokenizer(formatted_text, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
    return inputs


test_sentence = "Я боюсь собаков"
instruction = "Исправь ошибку и объясни, что было неправильно:"
suffix = "\n### Исправленный текст:\n"

#
inputs = preprocess_for_inference(test_sentence, instruction, suffix, tokenizer, max_length=256)

# Генерация ответа
model.eval()
with torch.no_grad():
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=300,
        num_return_sequences=1,
        temperature=0.4,
        top_p=0.9,
        top_k=80,
        do_sample=True
    )

# Декодирование и вывод
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print("Результат инференса:", generated_text)






#LOSS
# import pandas as pd
# from datasets import Dataset, DatasetDict
# from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
#
#
# file_path = "instructions_privet_c.xlsx"
# df = pd.read_excel(file_path)
#
#
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
#
#
# train_data = Dataset.from_pandas(train_df)
# valid_data = Dataset.from_pandas(test_df)
#
#
# model_path = './labels/checkpoint-1944'  # Путь к дообученной модели
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path).to('cpu')
#
#
# def preprocess(data, tokenizer):
#     prompts = data['Input']
#     responses = data['Output']
#     instruction = "Исправь ошибку и объясни, что было неправильно:"
#     suffix = "\n### Исправленный текст:\n"
#
#     combined = []
#     for prompt, response in zip(prompts, responses):
#         combined_text = f"### Инструкция:\n{instruction}\n\n### Входные данные:\n{prompt}\n\n### Ответ:\n{response}{suffix}"
#         combined.append(combined_text)
#
#     tokenized = tokenizer(combined, truncation=True, padding='max_length', max_length=256, return_tensors="pt")
#     labels = tokenized['input_ids'].clone()
#     labels[tokenized['attention_mask'] == 0] = -100
#
#     return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask'], 'labels': labels}
#
#
# dataset = DatasetDict({
#     'train': train_data,
#     'validation': valid_data,
# })
#
# dataset = dataset.map(
#     lambda d: preprocess(d, tokenizer),
#     batched=True
# )
#
#
# training_args = TrainingArguments(
#     output_dir='./labels/checkpoint-1944',
#     per_device_eval_batch_size=4,
#     logging_dir='./logs',
#     evaluation_strategy='no',
# )
#
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     eval_dataset=dataset['validation'],
#     tokenizer=tokenizer,
# )
#
#
# eval_results = trainer.evaluate(eval_dataset=dataset['validation'])
# train_results = trainer.evaluate(eval_dataset=dataset['train'])
#
#
# eval_loss = eval_results['eval_loss']
# train_loss = train_results['eval_loss']
#
# # Визуализация потерь
# datasets = ['Training', 'Validation']
# loss_values = [train_loss, eval_loss]
#
# plt.figure(figsize=(10, 5))
# plt.bar(datasets, loss_values, color=['blue', 'orange'])
# plt.xlabel('Dataset')
# plt.ylabel('Average Loss')
# plt.title('Comparison of Average Loss between Training and Validation Datasets')
# plt.show()





