import json
import os
import os.path as op


def write_json(obj, path):
    path = op.expanduser(path)
    with open(path, 'w') as f:
        json.dump(obj, f)


def read_json(path):
    path = op.expanduser(path)
    with open(path, 'r') as f:
        return json.load(f)


def create_dir_if_does_not_exist(path):
    path = op.expanduser(path)
    if not op.exists(path):
        os.makedirs(path)
