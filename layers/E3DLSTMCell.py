import torch
import torch.nn as nn

class EideticLSTMCell(nn.Module):
    # Eidetic3D LSTM Recurrent Network Cell
    def __init__(self, in_channels, out_channels, kernel_size):
        super(EideticLSTMCell, self).__init__()
        self.out_channels = out_channels
        padding = [None] * len(kernel_size)
        for i in range(len(kernel_size)):
            padding[i] = kernel_size[i] // 2
        self.conv_x = nn.Sequential(
            nn.Conv3d(in_channels, out_channels * 7, kernel_size=kernel_size, padding=padding),
        )
        self.conv_h = nn.Sequential(
            nn.Conv3d(out_channels, out_channels * 4, kernel_size=kernel_size, padding=padding),
        )
        self.conv_m = nn.Sequential(
            nn.Conv3d(out_channels, out_channels * 4, kernel_size=kernel_size, padding=padding),
        )
        self.conv_c = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )
        self.conv_o = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)
        self.layernorm = nn.LayerNorm(out_channels)

    def _attention(self, query, keys, values):
        # query -> [batch, channels, t1, h, w]
        # keys, values -> [batch, channels, t2, h, w]
        # return attn -> [batch, t1, h, w, channels]
        batch = query.shape[0]
        channel = query.shape[1]
        t = query.shape[2]
        h = query.shape[3]
        w = query.shape[4]

        query = torch.reshape(query, [batch, channel, -1]).permute([0, 2, 1])
        keys = torch.reshape(keys, [batch, channel, -1])
        values = torch.reshape(values, [batch, channel, -1]).permute([0, 2, 1])

        attn = torch.bmm(query, keys)
        attn = torch.softmax(attn, dim=2)
        attn = torch.bmm(attn, values)
        attn = torch.reshape(attn, [batch, t, h, w, channel])

        return attn



    def forward(self, x, h, c, m, c_his):
        # x,h,m -> [batch, channels, t, x, y]
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)
        x_i, x_g, x_r, x_i_prime, x_g_prime, x_f_prime, x_o = torch.split(x_concat, self.out_channels, dim=1)
        h_i, h_g, h_r, h_o = torch.split(h_concat, self.out_channels, dim=1)
        m_i, m_g, m_f, m_o = torch.split(m_concat, self.out_channels, dim=1)

        i_gate_c = torch.sigmoid(x_i + h_i)
        r_gate_c = torch.sigmoid(x_r + h_r)
        g_c = torch.tanh(x_g + h_g)
        recall = c.permute([0, 2, 3, 4, 1]) + self._attention(r_gate_c, c_his, c_his)
        c_next = i_gate_c * g_c + self.layernorm(recall).permute([0, 4, 1, 2, 3])

        i_gate_m = torch.sigmoid(x_i_prime + m_i)
        f_gate_m = torch.sigmoid(x_f_prime + m_f)
        g_m = torch.tanh(x_g_prime + m_g)
        m_next = i_gate_m * g_m + f_gate_m * m

        o_gate = torch.sigmoid(x_o + h_o + m_o + self.conv_c(c))
        h_next = o_gate * torch.tanh(self.conv_o(torch.cat([c_next, m_next],  dim=1)))

        return h_next, c_next, m_next


if __name__ == '__main__':
    net = EideticLSTMCell(in_channels=3, out_channels=8, kernel_size=[3, 5, 5])
    x = torch.rand([2, 3, 6, 20, 20])
    h = torch.rand([2, 8, 6, 20, 20])
    c = torch.rand([2, 8, 6, 20, 20])
    m = torch.rand([2, 8, 6, 20, 20])
    c_his = torch.rand([2, 8, 30, 20, 20])
    hn, cn, mn = net(x, h, c, m, c_his)
