import torch
import torch.nn as nn
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        """
        in_dim   -- input feature's channel dim
        activation    -- activation function type
        """
        super(Self_Attn, self).__init__()
        self.input_channel = in_dim
        self.activation = activation
        self.k = 8
        self.query = nn.Conv2d(self.input_channel, self.input_channel // self.k, kernel_size=1)
        self.key = nn.Conv2d(self.input_channel, self.input_channel // self.k, kernel_size=1)
        self.value = nn.Conv2d(self.input_channel, self.input_channel, kernel_size=1)
        self.h = nn.Conv2d(self.input_channel, self.input_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channel, width, height = x.size()

        q = self.query(x)
        q = q.view(batch_size, -1, width*height)
        print('q.shape',q.shape)
        k = self.key(x)
        k = k.view(batch_size, -1, width*height)
        print('k.shape',k.shape)
        s = torch.bmm(q.transpose(1,2), k)
        print('s.shape',s.shape)
        attention_map = self.softmax(s)
        print('attention_map.shape',attention_map.shape)
        h = self.h(x)
        h = h.view(batch_size, -1, width*height)
        print('h.shape',h.shape)
        attention_map_h = torch.bmm(h, attention_map.transpose(1,2))
        print('attention_map_h.shape',attention_map_h.shape)
        attention_map_h = attention_map_h.view(batch_size, -1, width, height)
        print('attention_map_h.shape',attention_map_h.shape)
        v = self.value(attention_map_h)
        print('v.shape',v.shape)
        out = x + self.gamma*v
        print('out.shape',out.shape)
        return out




        return None

image = np.random.randn(1, 64, 4, 4)
image = image.astype('float32')
image = torch.from_numpy(image)
model = Self_Attn(64, 'relu')
out = model(image)