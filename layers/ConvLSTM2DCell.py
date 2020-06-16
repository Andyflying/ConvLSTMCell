import torch
import torch.nn as nn

class ConvLSTM2D(nn.Module):
    # ConvLSTM2D Recurrent Network Cell
    def __init__(self, in_channels, out_channels, kernel_size, c_shape):
        super(ConvLSTM2D, self).__init__()
        self.out_channels = out_channels
        padding = [None] * len(kernel_size)
        for i in range(len(kernel_size)):
            padding[i] = kernel_size[i] // 2
        self.conv_x = nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size, padding=padding, bias=True)
        self.conv_h = nn.Conv2d(out_channels, out_channels * 4, kernel_size=kernel_size, padding=padding, bias=False)
        self.mul_c = nn.Parameter(torch.zeros([1, out_channels * 3, c_shape[0], c_shape[1]], dtype=torch.float32))


    def forward(self, x, h, c):
        # x -> [batch_size, channels, x, y]
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        i_x, f_x, c_x, o_x = torch.split(x_concat, self.out_channels, dim=1)
        i_h, f_h, c_h, o_h = torch.split(h_concat, self.out_channels, dim=1)
        i_c, f_c, o_c = torch.split(self.mul_c, self.out_channels, dim=1)
        i_t = torch.sigmoid(i_x + i_h + i_c * c)
        f_t = torch.sigmoid(f_x + f_h + f_c * c)
        c_t = torch.tanh(c_x + c_h)
        c_next = i_t * c_t + f_t * c
        o_t = torch.sigmoid(o_x + o_h + o_c * c_next)
        h_next = o_t * torch.tanh(c_next)
        return h_next, c_next

if __name__ == '__main__':
    x = torch.rand([2, 3, 20, 20])
    h = torch.rand([2, 8, 20, 20])
    c = torch.rand([2, 8, 20, 20])
    net = ConvLSTM2D(in_channels=3, out_channels=8, kernel_size=[5, 5], c_shape=c.shape[-2:])
    hn, cn = net(x, h, c)
