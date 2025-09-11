import torch, numpy as np
from muse_pipeline_downsampled import TrainingConfig, MuseTrainer

cfg = TrainingConfig()
trainer = MuseTrainer(cfg)
trainer.vae.eval()

# grab a small batch of your HR tiles (they're in [0,1] now)
lo, hi, _ = next(iter(trainer.val_loader))
x01 = hi.to(trainer.device).float()[:4]          # [0,1]
xm1 = x01 * 2 - 1                                 # [-1,1]

with torch.no_grad():
    rec01 = trainer.vae.decode(trainer.vae.encode(x01)[0])
    recm1 = trainer.vae.decode(trainer.vae.encode(xm1)[0])
    mse01 = torch.mean((rec01 - x01)**2).item()
    msem1 = torch.mean((recm1 - xm1)**2).item()

print("MSE when feeding [0,1]:", mse01)
print("MSE when feeding [-1,1]:", msem1)