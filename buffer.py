import torch
import numpy as np


class ReservoirBuffer:
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.x = []
        self.y = []
        self.task_ids = []

    def add_data(self, x, y, task_id):
        batch_size = x.shape[0]
        for i in range(batch_size):
            if self.num_seen_examples < self.buffer_size:
                self.x.append(x[i].cpu())
                self.y.append(y[i].cpu())
                self.task_ids.append(task_id)
            else:
                j = np.random.randint(0, self.num_seen_examples + 1)
                if j < self.buffer_size:
                    self.x[j] = x[i].cpu()
                    self.y[j] = y[i].cpu()
                    self.task_ids[j] = task_id
            self.num_seen_examples += 1

    def get_random_batch(self, batch_size):
        if len(self.x) == 0:
            return None, None, None

        real_batch_size = min(batch_size, len(self.x))
        indices = np.random.choice(len(self.x), real_batch_size, replace=False)

        batch_x = torch.stack([self.x[i] for i in indices]).to(self.device, non_blocking=True)
        batch_y = torch.stack([self.y[i] for i in indices]).to(self.device, non_blocking=True)
        batch_tid = torch.tensor([self.task_ids[i] for i in indices], dtype=torch.long).to(self.device,
                                                                                           non_blocking=True)

        return batch_x, batch_y, batch_tid