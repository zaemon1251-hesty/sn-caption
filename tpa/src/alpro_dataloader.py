"""
cite: https://github.com/THU-KEG/goal/blob/master/models/ALPRO/src/datasets/dataloader.py
"""

import torch


def move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        return [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        return {n: move_to_cuda(t) for n, t in batch.items()}
    return batch


def record_cuda_stream(batch):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream())
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t)


class PrefetchLoader:
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        loader_iter = iter(self.loader)
        self.preload(loader_iter)
        batch = self.next(loader_iter)
        while batch is not None:
            yield batch
            batch = self.next(loader_iter)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return

        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch)

    def next(self, it):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        return getattr(self.loader, name)


class InfiniteIterator:
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(iterable)

    def __iter__(self):
        while True:
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.iterable)
                batch = next(self.iterator)
            yield batch
