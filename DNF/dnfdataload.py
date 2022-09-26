import numpy as np
import torch
import pdb


class data_load:
    class Data:
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.y = data.astype(np.int64)

    def __init__(self, path, nclass=0):
        data, label = load_data_normalised(path, nclass)
        self.dt = self.Data(data)
        self.label = self.Data(label)


def get_data(path):
    data = np.load(path)

    x_vector = data['vector']
    label_vector = data['utt']
    x_vector = np.array(x_vector)
    label_vector = np.array(label_vector)



    return x_vector, label_vector


def load_data_normalised(path, n_filter):
    X, labels = get_data(path)
    return X, labels


def dataset_prepare(n_filter=0, train_path=None,test_path=None):
    # training set voxceleb_4k_speaker
    print("loading training data from %s" % train_path);
    train_data = data_load(train_path, n_filter)
    trn_tensor = torch.from_numpy(train_data.dt.x)
    trn_label_tensor = torch.from_numpy(train_data.label.y)
    train_dataset = torch.utils.data.TensorDataset(trn_tensor, trn_label_tensor)


    # testset: verify
    print("loading enrollment data from %s" % test_path);
    ver_data = data_load(test_path, 0)
    test_tensor = torch.from_numpy(ver_data.dt.x)

    test_label_tensor = torch.from_numpy(ver_data.label.y)
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_label_tensor)

    return train_dataset,  test_dataset



