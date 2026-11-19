import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path

ROOT = Path("e1_outputs/e1_handoff")
CACHE = ROOT / "cache"
DATA = ROOT / "data"

class E1Dataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

        self.tweet = np.load(CACHE / "account_tweet_embeddings.npy", mmap_mode="r")
        self.profile = np.load(CACHE / "account_profile_embeddings.npy", mmap_mode="r")
        self.struct = np.load(CACHE / "profile_numeric_boolean.npy", mmap_mode="r")
        self.neighbor = np.load(CACHE / "neighbor_aggregated_embeddings.npy", mmap_mode="r")
        self.labels = np.load(CACHE / "labels.npy", mmap_mode="r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]
        return (
            torch.tensor(self.tweet[i].copy(), dtype=torch.float32),
            torch.tensor(self.profile[i].copy(), dtype=torch.float32),
            torch.tensor(self.struct[i].copy(), dtype=torch.float32),
            torch.tensor(self.neighbor[i].copy(), dtype=torch.float32),
            torch.tensor(self.labels[i].copy(), dtype=torch.long)
        )


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size=4):
        assert batch_size % 2 == 0
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.per_class = batch_size // 2

        self.class_indices = [
            np.where(self.labels == c)[0].tolist()
            for c in [0, 1]
        ]

        assert min(map(len, self.class_indices)) >= self.per_class

    def __iter__(self):
        pools = [
            np.random.permutation(x).tolist()
            for x in self.class_indices
        ]
        steps = min(map(len, pools)) // self.per_class

        for step in range(steps):
            batch = []
            for pool in pools:
                start = step * self.per_class
                batch.extend(pool[start:start+self.per_class])
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return min(map(len, self.class_indices)) // self.per_class


with open(CACHE / "account_metadata.json") as f:
    metadata = json.load(f)

split_names = np.load(CACHE / "split_names.npy").astype(str)
labels = np.load(CACHE / "labels.npy")

assert len(metadata) == len(labels) == len(split_names)

train_idx = np.where(split_names == "train")[0]
dev_idx = np.where(split_names == "dev")[0]
test_idx = np.where(split_names == "test")[0]

train_dataset = E1Dataset(train_idx)
dev_dataset = E1Dataset(dev_idx)
test_dataset = E1Dataset(test_idx)

train_labels = labels[train_idx]
sampler = BalancedBatchSampler(train_labels, batch_size=4)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=sampler,
    num_workers=0
)

dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Train:", len(train_dataset), "batches:", len(train_loader))
print("Dev:", len(dev_dataset), "batches:", len(dev_loader))
print("Test:", len(test_dataset), "batches:", len(test_loader))
print("Train label counts:", np.bincount(train_labels))
print("DataLoader preparation: SUCCESS")
