import torch
import pytorch_lightning as pl
from model.model import PointerGenerator
from model.dataset import DatasetBase, collate_fn
from pl_version.pytorch_lightning_model import PlPointerGenerator
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# init the origin model
model = PointerGenerator(
    if_pointer=True,
    if_coverage=True,
    input_size=50,
    hidden_size=40,
    vocab_size=80200,
    batch_size=8,
    embw_size=50,
    attention_size=50,
    device=device
)
dataset = DatasetBase(
        input_ids_path="./dataset/ids.pkl",
        y_path="./dataset/y.pkl",
        vocab_path="./dataset/word_id.pkl",
        oov_words_path="./dataset/oov_words.pkl",
    )
loader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn,
)
pl_model = PlPointerGenerator(
    torch_model=model,
    lr=1e-3,
    batch_size=16,
    if_warm_up=False,
    LR_scheduler="step",
    optim_type="Adam",
    use_wandb=False,
    use_tensorboard=False,
    debug=True,
)

trainer = pl.Trainer(fast_dev_run=5,
                     accelerator="gpu",
                     devices=1,
                     default_root_dir="./check_points",
                     max_epochs=10,
                     )
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    trainer.fit(model=pl_model, train_dataloaders=loader)
print(prof.key_averages().table(sort_by="self_cpu_time_total"))