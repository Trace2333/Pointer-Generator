import torch
from model import PointerGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PointerGenerator(
    if_pointer=True,
    if_coverage=True,
    input_size=300,
    hidden_size=200,
    vocab_size=10000,
    batch_size=4,
    embw_size=300,
    attention_size=400,
    device=device
).to(device)
input_ids = [[1, 2, 3, 5, 67, 8, 9, 2, 34, 532, 63, 432, 23, 13],
             [1, 2, 3, 5, 67, 8, 9, 2, 34, 532, 63, 432, 23, 13],
             [1, 2, 3, 5, 67, 8, 9, 2, 34, 532, 63, 432, 23, 13],
             [1, 2, 3, 5, 67, 8, 9, 2, 34, 532, 63, 432, 23, 13]]

y = [[1, 2, 3, 5, 67, 8, 9, 2, 34, 532, 63, 432, 23, 13],
             [1, 2, 3, 5, 67, 8, 9, 2, 34, 532, 63, 432, 23, 13],
             [1, 2, 3, 5, 67, 8, 9, 2, 34, 532, 63, 432, 23, 13],
             [1, 2, 3, 5, 67, 8, 9, 2, 34, 532, 63, 432, 23, 13]]
vocab_out = model(input_ids, y)