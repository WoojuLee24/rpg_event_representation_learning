import numpy as np
import os
from os import listdir
from os.path import join
import torch


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


class NCaltech101(torch.utils.data.Dataset):
    def __init__(self, root, augmentation=False, tau=1):
        self.classes = 101
        self.file_list = listdir(root)

        self.files = []
        self.labels = []
        self.tau = tau
        self.augmentation = augmentation

        for i, c in enumerate(self.file_list):
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
        if self.tau > 1:
            events = events[events[:, 2] < 1 / self.tau]
            events[:, 2] = events[:, 2] * self.tau
        elif self.tau == -1:
            first_event = events.copy()
            second_event = events.copy()
            first_event[:, 2] = first_event[:, 2] / 2
            second_event[:, 2] = second_event[:, 2] / 2 + 1 / 2
            events = np.concatenate((first_event, second_event), axis=0)
        return events, label

