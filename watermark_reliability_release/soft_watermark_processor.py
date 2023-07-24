import torch
import hashlib
from transformers import LogitsProcessor
import numpy as np

def secret_number_generator(context, candidate):
    key = str(context) + str(candidate)
    np.random.seed(int(hashlib.sha256(bytes(key, encoding = 'ascii')).hexdigest()[-5:], 16))
    return np.random.random()

class soft_watermark_processor(LogitsProcessor):
    def __init__(self, sample_count, context_size, device):
        self.sample_count = sample_count
        self.context_size = context_size
        self.device = device

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        initial_sampling = torch.multinomial(torch.nn.functional.softmax(scores, dim = 1), num_samples = self.sample_count, replacement = True)
        secret_numbers = torch.zeros_like(initial_sampling, dtype=torch.float32)
        context = input_ids[:,-self.context_size:]
        batch_size = secret_numbers.shape[0]
        for batch_num in range(batch_size):
            for sample_num in range(self.sample_count):
                secret_numbers[batch_num,sample_num] = torch.tensor(secret_number_generator(context[batch_num].tolist(), initial_sampling[batch_num][sample_num].item()))

        chosen_secret_indices = secret_numbers.argmax(axis = 1)
        chosen_tokens = initial_sampling[torch.arange(batch_size), chosen_secret_indices]
        resulting_logits = torch.ones(scores.shape).to(self.device) * -float("inf")
        resulting_logits[torch.arange(batch_size).to(self.device), chosen_tokens] = 1
        return resulting_logits