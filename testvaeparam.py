import torch

ckpt = torch.load('/scratch/rnd-rojas/Manan/qv_results4/vae.best_srcc.pt', map_location='cpu')
params = ckpt['model'] if 'model' in ckpt else ckpt

for k in params.keys():
    print(k)