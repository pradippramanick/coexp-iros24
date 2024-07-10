from collections import Counter
import pandas
from sklearn.model_selection import train_test_split
from utils.ME_constants import action_prefix, Obs_prefix, Plan_prefix_template, test_only_tasks, label2id
from utils.ai2thor_utils import camel_case_split
from utils.serialization_utils import json_load

reflect_dataset = json_load('data/RoboFail_generated_w_SRL.json')
cf_dataset = json_load('data/CounterFactual_generated.json')


merged = []
reflect_minus_test_only = []


def convert_row_to_nl(row):
    task_id, task_type, premise_type, obs, hyp, L = row
    if premise_type == 'observation':
        obs = ', '.join(data['observation'])
        premise_conjunction = action_prefix + data['action'] + '. ' + Obs_prefix + obs + '.'
        return [task_id, task_type, premise_type, premise_conjunction, hyp, L]
    elif premise_type == 'plan':
        template = Plan_prefix_template
        template = template.replace('[TASK]', task_type)
        premise_conjunction = template + ', '.join(plan).lower() + '.'
        return [task_id, task_type, premise_type, premise_conjunction, hyp, L]
    else:
        raise ValueError(row)


# de-merge plan and obs hyps
i = 0
for data in reflect_dataset:
    task_id = data['task_id']
    label = data['label']
    L_plan_exp = data['L_plan_exp']
    L_obs_exp = data['L_obs_exp']
    hyp = data['explanation']
    plan = data['plan_until_failure']
    obs = data['observation']
    if 'comment' in data:
        if L_obs_exp == '' or L_plan_exp == '':
            print(data)
        task_type = task_id.split('-')[0]
        task_type = task_type[0].upper() + task_type[1:]
        task_type = ' '.join(camel_case_split(task_type)).lower()
        if task_type in test_only_tasks:
            reflect_minus_test_only.append(convert_row_to_nl([task_id, task_type, 'observation', obs, hyp, L_obs_exp]))
            reflect_minus_test_only.append(convert_row_to_nl([task_id, task_type, 'plan', plan, hyp, L_plan_exp]))
        else:
            merged.append(convert_row_to_nl([task_id, task_type, 'observation', obs, hyp, L_obs_exp]))
            merged.append(convert_row_to_nl([task_id, task_type, 'plan', plan, hyp, L_plan_exp]))

print('Total in reflect : ', len(reflect_minus_test_only) + len(merged))
print('Total in CF : ', len(cf_dataset)*2)
labels = [e[-1] for e in merged + reflect_minus_test_only]
print('Reflect counts: ', Counter(labels))

for data in cf_dataset:
    task_id = data['task_id']
    label = data['label']
    L_plan_exp = data['L_plan_exp']
    L_obs_exp = data['L_obs_exp']
    hyp = data['explanation']
    plan = data['plan_until_failure']
    obs = data['observation']
    task_type = task_id.split('-')[0]
    task_type = task_type[0].upper() + task_type[1:]
    task_type = ' '.join(camel_case_split(task_type)).lower()
    merged.append(convert_row_to_nl([task_id, task_type, 'observation', obs, hyp, L_obs_exp]))
    merged.append(convert_row_to_nl([task_id, task_type, 'plan', plan, hyp, L_plan_exp]))

merged_labels = [e[-1] for e in merged]
print('Test only examples', len(reflect_minus_test_only))
train, test = train_test_split(merged, test_size=.2, random_state=0, stratify=merged_labels)

print('Initial split')
print(len(train), len(test))
train_labels = [e[-1] for e in train]
test_labels = [e[-1] for e in test]
reflect_minus_test_only_labels = [e[-1] for e in reflect_minus_test_only]
print('Initial train,test,test-only')
print(Counter(train_labels), Counter(test_labels), Counter(reflect_minus_test_only_labels))
train, val = train_test_split(train, test_size=.1, random_state=0, stratify=train_labels)
val_labels = [e[-1] for e in val]
train_labels = [e[-1] for e in train]
print('After val split - train,val,test,test-only')
print(Counter(train_labels), Counter(val_labels), Counter(test_labels), Counter(reflect_minus_test_only_labels))

# dump into csvs
train = pandas.DataFrame(train, columns=["task_id", "task_type", "premise_type", "premise", "hypothesis", "label"])
val = pandas.DataFrame(val, columns=["task_id", "task_type", "premise_type", "premise", "hypothesis", "label"])
test = pandas.DataFrame(test + reflect_minus_test_only,
                        columns=["task_id", "task_type", "premise_type", "premise", "hypothesis", "label"])


train.to_csv('data/preprocessed/train.csv', encoding='utf-8', index=False, sep='\t')
val.to_csv('data/preprocessed/val.csv', encoding='utf-8', index=False, sep='\t')
test.to_csv('data/preprocessed/test.csv', encoding='utf-8', index=False, sep='\t')
print("Final split - train, val, test")
print(len(train), len(val), len(test))

print('\n Details')
print('Train - task type', train.value_counts('task_type'))
print('Val - task type', val.value_counts('task_type'))
print('Test - task type', test.value_counts('task_type'))
