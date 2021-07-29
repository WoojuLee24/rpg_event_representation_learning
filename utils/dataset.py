import numpy as np
from os import listdir
from os.path import join
import torch
import loris
from numpy.lib.recfunctions import structured_to_unstructured


def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    events[:,0] += x_shift
    events[:,1] += y_shift

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events

def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events


class NCaltech101:
    def __init__(self, root, augmentation=False):
        self.classes = listdir(root)

        self.files = []
        self.labels = []

        self.augmentation = augmentation

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)
        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)
        # normalizing time stamps
        events[:, 2] = events[:, 2] / events[:, 2].max(axis=0)
        return events, label


class NCARS(torch.utils.data.Dataset):
    def __init__(self, root, augmentation=False):
        self.classes = listdir(root)

        self.files = []
        self.labels = []

        self.augmentation = augmentation

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = self.files[idx]
        # events = np.load(f).astype(np.float32)
        with open(f, 'r') as f:
            d = f.readlines()
        events = loris.read_file(f)
        events = np.array(structured_to_unstructured(events, dtype=np.float32))
        # swap [t,x,y,p] to [x,y,t,p]
        events = events[:, [1, 2, 0, 3]]
        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)
        # normalizing time stamps
        events[:, 2] = events[:, 2] / events[:, 2].max(axis=0)
        return events, label
