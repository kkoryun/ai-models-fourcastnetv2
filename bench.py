from ai_models_fourcastnetv2.fourcastnetv2.sfnonet import FourierNeuralOperatorBlock
from ai_models_fourcastnetv2.fourcastnetv2.layers import RealFFT2, InverseRealFFT2, RealSHT2, InverseRealSHT2

import habana_frameworks.torch.gpu_migration
import habana_frameworks.torch.core as htcore

from torch import nn
import torch
import time
from functools import partial

# import habana_frameworks.torch.core as htcore

DEVICE = 'cpu'

num_layers = 12
embed_dim = 256
modes_lat = 120
modes_lon = 121
img_size = (721, 1440)
h = 120
w = 240
mlp_mode = 'distributed'
filter_type = 'non-linear'
dpr = [x.item() for x in torch.linspace(0, 0.0, num_layers)]

norm_layer0 = partial(
    nn.InstanceNorm2d,
    num_features=embed_dim,
    eps=1e-6,
    affine=True,
    track_running_stats=False,
)
norm_layer1 = norm_layer0

trans_down = RealSHT2(*img_size, lmax=modes_lat, mmax=modes_lon, grid="equiangular").float()
itrans_up = InverseRealSHT2(*img_size, lmax=modes_lat, mmax=modes_lon, grid="equiangular").float()
trans = RealSHT2(h, w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss").float()
itrans = InverseRealSHT2(h, w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss").float()

blocks = nn.ModuleList([])
for i in range(num_layers):
    first_layer = i == 0
    last_layer = i == num_layers - 1

    forward_transform = trans_down if first_layer else trans
    inverse_transform = itrans_up if last_layer else itrans

    inner_skip = "linear" if 0 < i < num_layers - 1 else None
    outer_skip = "identity" if 0 < i < num_layers - 1 else None
    mlp_mode = mlp_mode if not last_layer else "none"

    if first_layer:
        norm_layer = (norm_layer0, norm_layer1)
    elif last_layer:
        norm_layer = (norm_layer1, norm_layer0)
    else:
        norm_layer = (norm_layer1, norm_layer1)

    block = FourierNeuralOperatorBlock(
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type=filter_type,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=dpr[i],
        norm_layer=norm_layer,
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        inner_skip=inner_skip,
        outer_skip=outer_skip,
        mlp_mode=mlp_mode,
        compression=None,
        rank=128,
        complex_network=True,
        complex_activation='real',
        spectral_layers=3,
        checkpointing=False,
    )

    blocks.append(block)

blocks.to(DEVICE)

x = torch.rand(torch.Size([1, 256, 721, 1440])).to(DEVICE)

t_s = time.perf_counter_ns()
t_s1 = t_s
for i, blk in enumerate(blocks):
    if i == 11:
        pass
    x = blk(x)
    t_e = time.perf_counter_ns()
    t_t = (t_e-t_s) / 10**6
    t_s = t_e
    print(f"Block {i} processing time {t_t} ms")

print(f"Full processing time {(time.perf_counter_ns() - t_s1) / 10**6} ms")
