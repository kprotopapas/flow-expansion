from sklearn.datasets import make_swiss_roll, make_moons
from matplotlib import pyplot as plt
from maxentdiff.models import DiffusionModel

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from maxentdiff.solvers import VPSDE
from maxentdiff.trainers.adjoint_matching_trajectory import sample_trajectories_ddpm



class LightningDiffusion(LightningModule):
    def __init__(self, model: DiffusionModel):
        super().__init__()
        self.model = model

    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    

    def training_step(self, batch, batch_idx):
        x0, = batch
        t = torch.rand(x0.shape[0]).to(x0.device)
        alpha, sig = self.model.sde.get_alpha_sigma(t[:, None])
        eps = torch.randn(x0.shape).to(x0.device)

        xt = torch.sqrt(alpha) * x0 + sig * eps

        eps_pred = self(xt, t[:, None])

        loss = torch.mean((eps - eps_pred)**2) / 2.
        self.log('loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


network = nn.Sequential(
    nn.Linear(3, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
)

sde = VPSDE(0.1, 12)

model = DiffusionModel(network, sde, data_shape=(2,))

logger = TensorBoardLogger('tb_dir')
sworl, r = make_swiss_roll(n_samples=100000, noise=0.1)

dataset = torch.tensor(sworl, dtype=torch.float32)
dataset = torch.hstack((dataset[:, 0, None], dataset[:, 2, None]))

dl = DataLoader(TensorDataset(dataset), batch_size=128, shuffle=True)
pl_model = LightningDiffusion(model)

trainer = Trainer(max_epochs=10, logger=logger)
trainer.fit(pl_model, dl)
gpu = torch.device('cuda')

x0 = torch.randn((256, 2)).to(gpu)

trajs, ts = sample_trajectories_ddpm(model, x0, 1000, avoid_inf=0.0, sample_jumps=True)
trajs.shape
plt.scatter(dataset[:1000, 0], dataset[:1000, 1])
plt.scatter(trajs[:, 0, 0], trajs[:, 0, 1])
plt.scatter(trajs[:, -1, 0], trajs[:, -1, 1])
plt.show()
