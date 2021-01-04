import os
import pandas as pd
import pickle

def make_dict(root='./dataset/ModelNet40'):
    cls_dict = {}
    idx = 0
    dirs = os.listdir(root)

    for dir in dirs:
        if dir[0] == '.':
            continue
        cls_dict[dir] = idx
        idx += 1

    with open(os.path.join(root, 'cls_dict.pkl'), 'wb') as f:
        pickle.dump(cls_dict, f)

def make_csv(root='./dataset/ModelNet40', split='train'):
    df = pd.DataFrame(columns=['path', 'label'])
    dirs = os.listdir(root)

    with open(os.path.join(root, 'cls_dict.pkl'), 'rb') as f:
        cls_dict = pickle.load(f)

    for dir in dirs:
        if dir[0] == '.' or os.path.isdir(os.path.join(root, dir)) == False:
            continue

        path = os.path.join(root, dir)
        for filename in os.listdir(os.path.join(path, split)):
            assert filename[-3:] == 'off'
            df = df.append({'path': os.path.join(path, split, filename),
                            'label': cls_dict[dir]}, ignore_index=True)

    df.to_csv(f'./dataset/{split}_list.csv')


if __name__ == '__main__':

    if os.path.isfile('./dataset/ModelNet40/cls_dict.pkl') == False:
        make_dict()

    make_csv(split='train')
    make_csv(split='test')