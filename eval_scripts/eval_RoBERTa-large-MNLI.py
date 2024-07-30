import pandas
from tqdm import tqdm

from inference_roberta import RobertaNLI
from ME_constants import test_only_tasks

from sklearn.metrics import accuracy_score, f1_score, classification_report

dataset_name = 'test.csv'
dataset = pandas.read_csv('data/preprocessed/' + dataset_name, delimiter='\t')
dataset = dataset.reset_index()
gt_labels = []
pred_labels = []

model = RobertaNLI(model_name='FacebookAI/roberta-large-mnli')


for i, data in tqdm(dataset.iterrows()):
    gt_label = data['label']
    pred = model.forward(data['premise'], data['hypothesis'], return_argmax=True)['label']
    pred_labels.append(pred)
    gt_labels.append(gt_label)
assert len(gt_labels) == len(pred_labels)
acc = accuracy_score(gt_labels, pred_labels)
f1_weighted = f1_score(gt_labels, pred_labels, average='weighted')
print(len(gt_labels), acc, f1_weighted)


print('Overall classification')
print(classification_report(gt_labels, pred_labels))
