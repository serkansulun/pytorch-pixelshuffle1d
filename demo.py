import torch
from torch import nn
import torch.nn.functional as F

from pixelshuffle1d import PixelShuffle1D, PixelUnshuffle1D

def conv1d_same(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    # 1D Convolution which does not change input size
    # "same" padding
    return torch.nn.Conv1d(in_channels, out_channels, kernel_size, 
                            padding=(kernel_size-1)//2, bias=bias, dilation=dilation)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
channel_len = 2
sample_len = 44100

x = torch.rand(batch_size, channel_len, sample_len)     # input

scale_factor = 2

# Use pixelshuffler as a module
pixel_upsample = PixelShuffle1D(scale_factor)
pixel_downsample = PixelUnshuffle1D(scale_factor)

# Check if PixelUnshuffle1D is the inverse of PixelShuffle1D
x_up = pixel_upsample(x)
x_up_down = pixel_downsample(x_up)

if torch.all(torch.eq(x, x_up_down)):
    print('Inverse module works.')

# Try to learn linear upsampling using PixelShuffle1D and CNN
t = nn.functional.interpolate(x, scale_factor=scale_factor, mode='linear', align_corners=False)    # target

n_conv_ch = 512
kernel_conv = 5

net = nn.Sequential(
                    pixel_upsample,
                    conv1d_same(channel_len//scale_factor, n_conv_ch, kernel_conv),
                    nn.ReLU(),
                    conv1d_same(n_conv_ch, channel_len, kernel_conv)
                    )

x = x.to(device)
t = t.to(device)
net = net.to(device)

loss_func = nn.MSELoss()
optim = torch.optim.Adam(net.parameters(), lr=1e-5)

for _ in range(10000):
    optim.zero_grad()
    y = net(x)   # run generator    
    loss = loss_func(y, t)     # mean square loss
    loss.backward()
    optim.step()
    print('Loss: {:.2e}'.format(loss))
