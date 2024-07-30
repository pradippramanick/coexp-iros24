root_dir = '/PATH_TO_YOUR_ROOT_DIR/'
thor_task_dir = root_dir + 'thor_tasks'
state_summary_dir = root_dir + 'state_summary'
action_prefix = 'Robot attempts to '
action_suffix = ' and fails.'
Obs_prefix = 'Robot sees '
Plan_prefix_template = 'To [TASK], robot has this plan - '
test_only_tasks = ['make salad', 'warm water', 'store egg']
label2id = {
    "C": 0,
    "N": 1,
    "E": 2
}
label_set = list(label2id.keys())
