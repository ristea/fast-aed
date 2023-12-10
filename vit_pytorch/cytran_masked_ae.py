from einops import rearrange
from torch import nn, einsum
import functools
import torch
from torch.nn import Conv2d

from networks.mask_in import MaskIn


class Encoder(nn.Module):
    def __init__(self, input_nc, ngf=16, norm_layer=nn.BatchNorm2d, n_downsampling=3):
        super(Encoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        self.down_sampling = nn.Sequential(*model)

    def forward(self, input):
        intermed_results = []
        for name, layer in self.down_sampling.named_children():
            input = layer(input)
            if isinstance(layer, Conv2d):
                intermed_results.append(input)
        return input, intermed_results


class Decoder(nn.Module):
    def __init__(self, output_nc, ngf=16, norm_layer=nn.BatchNorm2d, n_downsampling=3):
        super(Decoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.model = nn.ModuleList()
        mult = 2 ** n_downsampling
        self.model.append(nn.Sequential(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                           kernel_size=3, stride=2,
                                                           padding=1, output_padding=(1, 0),
                                                           bias=use_bias),
                                        norm_layer(int(ngf * mult / 2)),
                                        nn.ReLU(True)))
        for i in range(1, n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            self.model.append(nn.Sequential(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                               kernel_size=3, stride=2,
                                                               padding=1, output_padding=1,
                                                               bias=use_bias),
                                            norm_layer(int(ngf * mult / 2)),
                                            nn.ReLU(True)))
        self.model.append(nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)))

    def forward(self, input, inter_results):
        out1 = self.model[0](input + inter_results[-1])
        out2 = self.model[1](out1 + inter_results[-2])
        out = self.model[2](out2 + inter_results[-3])
        # out = self.model[3](out+inter_results[-4])
        return out + inter_results[-4], input, out1, out2


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_q = DepthWiseConv2d(dim, inner_dim, 3, padding=padding, stride=1, bias=False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding=padding, stride=kv_proj_stride, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, y=y)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head=64, mlp_mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride, heads=heads,
                                       dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CyTranAE(nn.Module):
    def __init__(self, input_nc, output_nc, n_downsampling, depth, heads, proj_kernel=3,
                 mlp_mult=4, dropout=0., ngf=16, pretraining=False, masking=False):

        super().__init__()
        dim = (2 ** n_downsampling) * ngf
        if masking:
            self.mask_layer = MaskIn(8, 0.4)
        self.conv_encoder = Encoder(input_nc=input_nc, ngf=ngf, n_downsampling=n_downsampling)
        self.transformer = Transformer(dim=dim, proj_kernel=proj_kernel, kv_proj_stride=2, depth=depth, heads=heads,
                                       mlp_mult=mlp_mult, dropout=dropout)
        self.decoder = Decoder(output_nc=output_nc, n_downsampling=n_downsampling)
        self.small = nn.Sequential(
            nn.AdaptiveMaxPool2d((2, 2)),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.medium = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 4))
        )

        self.high = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((16, 16))
        )
        self.out_image = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)
        self.pretraining = pretraining
        self.masking = masking

    def forward(self, img):
        if self.masking:
            img = self.mask_layer(img)
        x_enc, inter_results = self.conv_encoder(img)
        x = self.transformer(x_enc)
        out, _, _, _ = self.decoder(x, inter_results)
        if not self.pretraining:
            small = self.small(out)
            medium = self.medium(out)
            high = self.high(out)
            return small, medium, high
        else:
            return self.out_image(out)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CyTranAE(input_nc=3, output_nc=1, n_downsampling=4, depth=5, heads=5).to(device)
    tensor = torch.zeros((2, 3, 640, 360))
    model(tensor.to(device))
