import io
import json
import os
import pickle
import sys


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "scene_graph":
            renamed_module = "reflect.scene_graph"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)


def json_load(path):
    return json.load(open(path))


def json_dump(obj, path):
    return json.dump(obj, open(path, 'w'), indent=1)


def get_local_minimal_sg(idx, local_sg_dir):
    # -1 because local graphs are indexed from 0, frames from 1
    path = os.path.join(local_sg_dir, 'local_sg_' + str(idx + 1) + '.pkl')
    if os.path.exists(path):
        sg = renamed_load(open(path, 'rb'))
    else:
        # Back off to idx, because of the inconsistent numbering formats in Reflect
        path = os.path.join(local_sg_dir, 'local_sg_' + str(idx) + '.pkl')
        if os.path.exists(path):
            sg = renamed_load(open(path, 'rb'))
        else:
            print('\n\n WARNING !!!!! Unable to get gt graph. check files. \nPath:', path)
            exit()
    # Convert to minimal SG
    try:
        del sg.event
    except:
        pass
    for i in range(len(sg.nodes)):
        #print(sg.nodes[i].name)
        del sg.nodes[i].pcd
        del sg.nodes[i].depth
    return sg
