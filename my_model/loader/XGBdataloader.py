import numpy as np
from torch.utils.data import Dataset, DataLoader

class XGBLoader(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode

        # (16992, 307, 3)
        self.data = np.load(self.args.data_path)['data'][:,:,0]
        # print(self.data)
        # print(self.data.size)
        # print(self.data.shape)
        # self.seq_len = args.seq_len
        self.seq_len = 12

        if self.mode == 'train':
            self.data = self.data[:40*288]
        elif self.mode == 'val':
            # 9
            self.data = self.data[40*288:-9*288]
        elif self.mode == 'test':
            # 10
            self.data = self.data[-(9*288+self.seq_len):]
        else:
            raise Exception('mode must be \'train\', \'val\' or \'test\'')

    def __getitem__(self, index):
        sample = []

        for i in range(self.seq_len+1):
            sample.append(self.data[index + i])
        return sample

    def __len__(self):
        return (len(self.data) - self.seq_len)