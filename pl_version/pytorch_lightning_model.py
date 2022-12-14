import os
import torch.nn
import torch
import wandb

from rouge import Rouge

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class PlPointerGenerator(pl.LightningModule):
    def __init__(self, torch_model, lr, batch_size, if_warm_up, LR_scheduler, optim_type, use_wandb=False, use_tensorboard=False, debug=False):
        """Init the pl model"""
        super(PlPointerGenerator, self).__init__()
        self.model = torch_model
        self.lr = lr
        self.batch_size = batch_size
        self.optim = optim_type
        self.if_warm_up = if_warm_up
        self.LR = LR_scheduler
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.lossfun = None
        self.debug = debug
        self.loss_function_init()
        self.wandb_init()
        self.rouge = Rouge()

    def training_step(self, batch, batch_idx):
        """batch : (articles, oov_words, abstracts, max_oov_nums)"""
        article_ids, oov_words, abstracts_ids, max_oov_nums, seq_lens = batch
        model_out = self.model(article_ids, oov_words, abstracts_ids, max_oov_nums, seq_lens)
        #loss = self.lossfun(model_out[:, 1:, :].permute(0, 2, 1), abstracts_ids)
        loss = self.loss_as_nll(model_out[:, 1:, :].permute(0, 2, 1), abstracts_ids)

        # metrics compute
        scores = self.rouge_compute(model_out, abstracts_ids)
        r1_f = scores["rouge-1"]["f"]
        r2_f = scores["rouge-2"]["f"]
        rl_f = scores["rouge-l"]["f"]

        if self.debug is not None and self.debug is True and self.use_wandb is True:
            wandb.log({"loss_gen": loss.item()})
            wandb.log({"Rouge1": r1_f})
            wandb.log({"Rouge2": r2_f})
            wandb.log({"RougeL": rl_f})
            # debug
            """for name, parms in self.model.named_parameters():  # debug时使用，可视化每一个层的grad与weight
                wandb.log({f"{name} Weight:": torch.mean(parms.data)})
                if parms.grad is not None:
                    wandb.log({f"{name} Grad_Value:": torch.mean(parms.grad)})"""

        return loss

    def validation_step(self, batch, batch_idx):
        article_ids, oov_words, abstracts_ids, max_oov_nums, seq_lens = batch
        model_out = self.model.evaluation(article_ids, abstracts_ids, oov_words, max_oov_nums, seq_lens)
        # loss = self.lossfun(model_out[:, 1:, :].permute(0, 2, 1), abstracts_ids)
        #loss = self.loss_as_nll(model_out[:, 1:, :].permute(0, 2, 1), abstracts_ids)

        # metrics compute
        scores = self.rouge_compute(model_out, abstracts_ids)
        r1_f = scores["rouge-1"]["f"]
        r2_f = scores["rouge-2"]["f"]
        rl_f = scores["rouge-l"]["f"]

        if self.debug is not None and self.debug is True and self.use_wandb is True:
            #wandb.log({"Eval-loss_gen": loss.item()})
            wandb.log({"Eval-Rouge1": r1_f})
            wandb.log({"Eval-Rouge2": r2_f})
            wandb.log({"Eval-RougeL": rl_f})

        return

    def test_step(self, batch, batch_idx):
        return

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return

    def forward(self, x):
        return self.model(x)

    """def configure_callbacks(self):
        callbacks = EarlyStopping(monitor="loss", mode='min')
        return [callbacks]"""

    def configure_optimizers(self):
        if self.optim is "Adam":
            optimizer = torch.optim.Adam(self.parameters(), self.lr)
        elif self.optim is "SGD":
            optimizer = torch.optim.SGD(self.parameters(), self.lr)
        elif self.optim is "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), self.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
        return ([optimizer], [scheduler])

    def wandb_init(self):
        if self.use_wandb is True:
            wandb.login(host="http://47.108.152.202:8080",
                key="local-86eb7fd9098b0b6aa0e6ddd886a989e62b6075f0")
            wandb.init(project="Pointer-Generator")
            os.system("wandb online")
        if self.use_wandb is False:
            os.system("wandb offline")
    def loss_function_init(self):
        self.lossfun = torch.nn.NLLLoss()
        return

    def loss_as_nll(self, p, target):
        loss_before = torch.gather(p, 1, target.unsqueeze(1)).squeeze()
        loss = torch.sum(-torch.log(loss_before + 1e-12)) / self.model.batch_size
        return loss

    def rouge_compute(self, pred, y):
        pred_ids = pred[:, 1:, :].argmax(-1).tolist()
        y_ids = y.tolist()

        f1s = []
        f2s = []

        for sen in pred_ids:
            f1 = ""
            for i in sen:
                f1 += str(i) + " "
            f1s.append(f1)

        for sen in y_ids:
            f2 = ""
            for i in sen:
                f2 += str(i) + " "
            f2s.append(f2)

        scores = self.rouge.get_scores(f1s, f2s, avg=True)
        return scores