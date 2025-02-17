from torch.utils.data import Dataset


class IndianPinesDataset(Dataset):
    def __init__(self, data):
        self.original = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
