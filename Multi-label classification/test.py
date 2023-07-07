import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss, average_precision_score, label_ranking_loss
from transformers import EvalPrediction

dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
# 12-layer, 768-hidden, 12-heads, 110M parameters
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_data(examples):
  # take a batch of texts
  text = examples["Tweet"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding

def multi_label_metrics(predictions, labels, threshold=0.5):

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    print(probs)

    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    print(y_pred)

    # predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    # print(predicted_labels)
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = average_precision_score(y_true, y_pred)
    ham = hamming_loss(y_true, y_pred)
    rl = label_ranking_loss(y_true, y_pred)
    # return as dictionary
    metrics = {'HL': ham,
               'RL': rl,
               'miF1': f1_micro_average,
               'maF1': f1_macro_average,
               'AP': accuracy,
               'roc_auc': roc_auc}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


#main start
encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
encoded_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained("bert-finetuned-sem_eval-english/checkpoint-4000", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)
batch_size = 8
metric_name = "miF1"

args = TrainingArguments(
    f"bert-finetuned-sem_eval-english",
    evaluation_strategy = "steps",
    save_strategy = "steps",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    save_total_limit = 5,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    # validate dataset
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

example = encoded_dataset['validation'][0]
print(example)
# print(tokenizer.decode(example['input_ids']))
# print(example['labels'])


# predictions = trainer.predict(encoded_dataset["validation"])
# print(predictions)


def multi_classification(par):
    text = ' '.join(par)
    encoding = tokenizer(text, return_tensors="pt")
    encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}
    outputs = trainer.model(**encoding)
    logits = outputs.logits
    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    return probs.tolist()

def Bert_pre(aspects):
    text = list(map(lambda x: x.strip(), aspects))
    chunks = [text[x:x+100] for x in range(0, len(text), 100)]
    preditions = list(map(multi_classification, chunks))
    fin_predictions = [sum(i)/len(preditions) for i in zip(*preditions)]
    return fin_predictions



# predictions = np.zeros(probs.shape)
# predictions[np.where(probs >= 0.5)] = 1
# turn predicted id's into actual label names
# predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]

# print(predicted_labels)