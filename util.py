
def read_data(task_name):
    if task_name == 'task1':
        with open(os.path.join(ANNOTATION_DIR, 'test_data.json'), 'r') as file:
            content = file.readlines()
        content = list(map(json.loads, content))
        return content
    elif task_name == 'task2':
        with open(os.path.join(ANNOTATION_DIR, 'task_2.json'), 'r') as file:
            content = file.readlines()
        content = list(map(json.loads, content))
        return content
