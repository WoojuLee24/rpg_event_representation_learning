import numpy as np
import os
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


class NCaltech101(torch.utils.data.Dataset):
    def __init__(self, root, augmentation=False):
        self.classes = 101
        self.file_list = listdir(root)

        self.files = []
        self.labels = []

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
        # with open(f, 'r') as f:
        #     d = f.readlines()
        data = loris.read_file(f)
        events = np.array(structured_to_unstructured(data['events'], dtype=np.float32))
        # events = np.array(data['events'], dtype=np.float32)

        # swap [t,x,y,p] to [x,y,t,p]
        events = events[:, [1, 2, 0, 3]]
        # if self.augmentation:
        #     events = random_shift_events(events, resolution=(240, 304))
        #     events = random_flip_events_along_x(events, resolution=(240, 304))
        # normalizing time stamps
        events[:, 2] = events[:, 2] / events[:, 2].max(axis=0)
        return events, label


class DVSGesture(torch.utils.data.Dataset):
    def __init__(self, root, augmentation=False):
        self.classes = 11
        self.files = []
        self.labels = []
        self.augmentation = augmentation

        # for i, c in enumerate(self.classes):
        #     new_files = [join(root, c, f) for f in listdir(join(root, c))]
        #     self.files += new_files
        #     self.labels += [i] * len(new_files)

        for path, dirs, files in os.walk(root):
            dirs.sort()
            for file in files:
                if file.endswith("npy"):
                    self.files.append(path + "/" + file)
                    self.labels.append(int(file[:-4]))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        events = np.load(self.files[idx])
        # events[:, 3] *= 1000  # convert from ms to us
        label = self.labels[idx]

        # swap [x,y,p,t] to [x,y,t,p]
        events = events[:, [0, 1, 3, 2]]
        # if self.augmentation:
        #     events = random_shift_events(events, resolution=(240, 304))
        #     events = random_flip_events_along_x(events, resolution=(240, 304))
        # normalizing time stamps
        events[:, 2] = events[:, 2] / events[:, 2].max(axis=0)
        H = events[:, 1].max(axis=0)
        W = events[:, 0].max(axis=0)
        events = events.astype(np.float32)
        return events, label

# class DVSGesture(torch.utils.data.Dataset):
#
#     _GESTURE_MAPPING_FILE = "gesture_mapping.csv"
#     _TRAIN_TRIALS_FILE = "trials_to_train.txt"
#     _TEST_TRIALS_FILE = "trials_to_test.txt"
#
#     _LABELS_DTYPE = np.dtype(
#         [
#             ("event", np.uint8),
#             ("start_time", np.uint32),  # In microsecond
#             ("end_time", np.uint32),
#         ]
#     )
#     _GESTURE_MAP = {}
#
#     def __init__(self, root, shuffle=True, train=True, augmentation=False):
#
#         # Read gestures mapping file
#         parsed_csv = np.genfromtxt(
#             os.path.join(root, self._GESTURE_MAPPING_FILE),
#             delimiter=",",
#             skip_header=1,
#             dtype=None,
#             encoding="utf-8",
#         )
#         gestures, indexes = list(zip(*parsed_csv))
#         self._GESTURE_MAP = dict(zip(indexes, gestures))
#
#         if train:
#             path = os.path.join(root, self._TRAIN_TRIALS_FILE)
#         else:
#             path = os.path.join(root, self._TEST_TRIALS_FILE)
#
#         with open(path, "r") as f:
#             self.files = map(lambda d: os.path.join(root, d.rstrip()), f.readlines())
#         self.files = list(filter(lambda f: os.path.isfile(f), self.files))
#
#         if shuffle:
#             np.random.shuffle(self.files)
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         """
#         returns events and label, loading events from aedat
#         :param idx:
#         :return: x,y,t,p,  label
#         """
#         events = np.load(self.files[idx])
#         # events[:, 3] *= 1000  # convert from ms to us
#         label = self.labels[idx]
#
#         # swap [x,y,p,t] to [x,y,t,p]
#         events = events[:, [0, 1, 3, 2]]
#         # if self.augmentation:
#         #     events = random_shift_events(events, resolution=(240, 304))
#         #     events = random_flip_events_along_x(events, resolution=(240, 304))
#         # normalizing time stamps
#         events[:, 2] = events[:, 2] / events[:, 2].max(axis=0)
#         return events, label


class NMNIST(torch.utils.data.Dataset):
    def __init__(self, root, augmentation=False, first_saccade_only=False):
        self.classes = 10
        self.root = root
        self.files = []
        self.labels = []
        self.first_saccade_only = first_saccade_only
        self.augmentation = augmentation

        for i, c in enumerate(listdir(root)):
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
        events = self._read_dataset_file(self.files[idx])
        label = self.labels[idx]
        # normalizing time stamps
        events[:, 2] = events[:, 2] / events[:, 2].max(axis=0)
        events[:, 3] = events[:, 3] * 2 - 1
        # if self.augmentation:
        #     events = random_shift_events(events, resolution=(28, 28))
        #     events = random_flip_events_along_x(events, resolution=(28, 28))

        return events, label


    def _read_dataset_file(self, filename):
        f = open(filename, "rb")
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()
        raw_data = raw_data.astype(int)

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_ts = (
            ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        )

        # Process time stamp overflow events
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts[overflow_index:] += time_increment

        # Everything else is a proper td spike
        td_indices = np.where(all_y != 240)[0]

        if self.first_saccade_only:
            td_indices = np.where(all_ts < 100000)[0]

        events = np.column_stack(
            (
                all_x[td_indices],
                all_y[td_indices],
                all_ts[td_indices],
                all_p[td_indices],
            )
        )
        return events.astype(np.float32)
