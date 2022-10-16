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
    input_size=300,
    hidden_size=200,
    vocab_size=80200,
    batch_size=8,
    embw_size=300,
    attention_size=400,
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
    batch_size=8,
    if_warm_up=False,
    LR_scheduler="step",
    optim_type="Adam",
    use_wandb=True,
    use_tensorboard=False,
    debug=True,
)

trainer = pl.Trainer(fast_dev_run=False,
                     accelerator="gpu",
                     devices=1,
                     default_root_dir="./check_points",
                     limit_train_batches=128,
                     max_epochs=10
                     )
trainer.fit(model=pl_model, train_dataloaders=loader)
