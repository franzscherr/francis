import pickle as pkl


def pickle(path, data):
    with open(path, 'wb') as f:
        return pkl.dump(data, path)


def depickle(path):
    with open(path, 'rb') as f:
        return pkl.load(f)
