from torch.utils.data.distributed import DistributedSampler
import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

class BalancedDistributedSampler(DistributedSampler):
    def __init__(self,dataset, positive_indices, negative_indices, batch_size, num_replicas=None, rank=None, shuffle=False, seed=0, drop_last=False):
        super().__init__(dataset)
        self.dataset = dataset
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # Calculate the number of positive samples per replica and adjust for drop_last
        self.num_samples_per_replica = len(self.positive_indices) // self.num_replicas
        if self.drop_last is True:
            self.num_samples_per_replica = (self.num_samples_per_replica // (self.batch_size // 2)) * (self.batch_size // 2)

        # The total samples for each replica should be double the positive samples because we have equal number of negative samples
        self.total_samples_per_replica = self.num_samples_per_replica * 2
        self.total_samples = self.total_samples_per_replica * self.num_replicas
        self.num_batches = self.num_samples_per_replica // (self.batch_size // 2)
        print(f'rank: {rank}, '
              f'num_batches: {self.num_batches}, '
              f'self.total_samples: {self.total_samples}, '
              f'self.total_samples_per_replica: {self.total_samples_per_replica}, '
              f'num_samples_per_replica: {self.num_samples_per_replica}, '
              f'num_replicas: {num_replicas}')

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        pos_indices = torch.tensor(self.positive_indices)
        neg_indices = torch.tensor(self.negative_indices)

        if self.shuffle:
            pos_indices = pos_indices[torch.randperm(len(pos_indices), generator=g)]
            neg_indices = neg_indices[torch.randperm(len(neg_indices), generator=g)]

        # Ensure each process gets a unique subset of positive samples
        pos_indices_split = torch.chunk(pos_indices, self.num_replicas)
        pos_indices_local = pos_indices_split[self.rank]

        if self.drop_last:
            pos_indices_local = pos_indices_local[:self.num_samples_per_replica]

        num_pos_samples = len(pos_indices_local)
        num_neg_samples = num_pos_samples

        if num_neg_samples > len(neg_indices):
            raise ValueError(f"Cannot sample {num_neg_samples} negative samples from only {len(neg_indices)} available.")

        neg_indices_local = torch.chunk(neg_indices[torch.randperm(len(neg_indices), generator=g)][:(self.num_samples_per_replica*self.num_replicas)], self.num_replicas)
        neg_indices_local = neg_indices_local[self.rank]
        indices = []
        # print(f'rank: {self.rank}, num_batches: {num_pos_samples // (self.batch_size // 2)}, num_pos_samples: {num_pos_samples}')
        if self.drop_last:
            for i in range(num_pos_samples // (self.batch_size // 2)):
                batch_pos = pos_indices_local[i * (self.batch_size // 2):(i + 1) * (self.batch_size // 2)]
                batch_neg = neg_indices_local[i * (self.batch_size // 2):(i + 1) * (self.batch_size // 2)]
                batch = torch.cat([batch_pos, batch_neg])
                batch = batch[torch.randperm(len(batch), generator=g)]
                indices.extend(batch.tolist())
        else:
            num_batches = len(pos_indices_local) // (self.batch_size // 2)
            for i in range(num_batches):
                batch_pos = pos_indices_local[i * (self.batch_size // 2):(i + 1) * (self.batch_size // 2)]
                batch_neg = neg_indices_local[i * (self.batch_size // 2):(i + 1) * (self.batch_size // 2)]
                batch = torch.cat([batch_pos, batch_neg])
                batch = batch[torch.randperm(len(batch), generator=g)]
                indices.extend(batch.tolist())

            remaining_pos = pos_indices_local[num_batches * (self.batch_size // 2):]
            remaining_neg = neg_indices_local[num_batches * (self.batch_size // 2):]
            if len(remaining_pos) > 0 and len(remaining_neg) > 0:
                remaining_batch = torch.cat([remaining_pos, remaining_neg])
                remaining_batch = remaining_batch[torch.randperm(len(remaining_batch), generator=g)]
                indices.extend(remaining_batch.tolist())
        # print(f'indices: {indices}, rank: {self.rank}')
        return iter(indices)

    def __len__(self):
        return self.total_samples_per_replica

    def set_epoch(self, epoch):
        self.epoch = epoch