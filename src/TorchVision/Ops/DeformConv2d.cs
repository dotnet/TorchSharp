using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using TorchVision.Modules;
using static TorchSharp.torch;

#nullable enable
namespace TorchVision
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            public static Modules.DeformConv2d DeformConv2d()
            {
                throw new NotImplementedException();
                //return new DeformConv2d();
            }
        }
    }

    namespace Modules
    {
        //https://github.com/dotnet/TorchSharp/issues/1472
        public class DeformConv2d : torch.nn.Module<Tensor, Tensor>
        {
            /*
             *
             *import torch
               import torch.nn as nn
               import torch.nn.functional as F
               
               class DeformConv2d(nn.Module):
                   def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
                       super(DeformConv2d, self).__init__()
               
                       self.in_channels = in_channels
                       self.out_channels = out_channels
                       self.kernel_size = (kernel_size, kernel_size)
                       self.stride = (stride, stride)
                       self.padding = (padding, padding)
               
                       self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
                       
                       self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
               
                       self.reset_parameters()
               
                   def reset_parameters(self):
                       nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                       if self.bias is not None:
                           nn.init.constant_(self.bias, 0)
               
                   def forward(self, x, offset):
                       
                       N, _, H_in, W_in = x.size()
                       C_out, C_in, Kh, Kw = self.weight.size()
                       H_out = (H_in + 2 * self.padding[0] - Kh) // self.stride[0] + 1
                       W_out = (W_in + 2 * self.padding[1] - Kw) // self.stride[1] + 1
                       
                       
                       p_x = torch.arange(-(Kw - 1) // 2, (Kw - 1) // 2 + 1)
                       p_y = torch.arange(-(Kh - 1) // 2, (Kh - 1) // 2 + 1)
                       p_x, p_y = torch.meshgrid(p_x, p_y, indexing='ij')
                       p = torch.cat([p_x.flatten(), p_y.flatten()], 0).view(1, 2 * Kh * Kw, 1, 1).to(x.device, x.dtype)
               
                       g_y = torch.arange(0, H_out * self.stride[0], self.stride[0])
                       g_x = torch.arange(0, W_out * self.stride[1], self.stride[1])
                       g_x, g_y = torch.meshgrid(g_x, g_y, indexing='ij')
                       grid = torch.cat([g_x.flatten(), g_y.flatten()], 0).view(1, 2, H_out, W_out).to(x.device, x.dtype)
                       grid = grid.repeat(N, 1, 1, 1)
               
                       p = p.view(1, 2, Kh * Kw, 1, 1)
                       grid = grid.unsqueeze(2)
                       offset = offset.view(N, 2, Kh * Kw, H_out, W_out)
               
                       vgrid = grid + p + offset
                       
                       vgrid_x = 2.0 * vgrid[:, 0, ...] / max(W_in - 1, 1) - 1.0
                       vgrid_y = 2.0 * vgrid[:, 1, ...] / max(H_in - 1, 1) - 1.0
                       
                       normalized_grid = torch.stack([vgrid_x, vgrid_y], dim=-1)
               
                       sampled_features = F.grid_sample(
                           x.unsqueeze(2).expand(-1, -1, Kh * Kw, -1, -1).reshape(N * C_in, Kh * Kw, H_in, W_in),
                           normalized_grid.view(N * C_in, Kh * Kw, H_out, W_out, 2),
                           mode='bilinear', padding_mode='zeros', align_corners=False
                       ).view(N, C_in, Kh * Kw, H_out, W_out)
               
                       output = torch.einsum('nikhw,oik->nohw', sampled_features, self.weight.view(C_out, C_in, Kh * Kw))
               
                       if self.bias is not None:
                           output += self.bias.view(1, -1, 1, 1)
               
                       return output
             */
            private Parameter? bias;
            private Parameter weight;
            private Conv2d offset_conv;
            private bool? use_bias;
            private int kernel_size;
            private long[] strides;
            private long[] padding;
            private long[] dilation;
            private long groups;
            protected internal DeformConv2d(int in_channels, int out_channels, int kernel_size, int stride=1, int padding=1, int dilation=1, int groups=1, bool? bias=false) : base(nameof(DeformConv2d))
            {
                this.strides = new long[] { stride, stride };
                this.padding= new long[] { padding,padding};
                this.dilation= new long[] { dilation,dilation};
                this.groups = groups;

                use_bias = bias;
                this.kernel_size = kernel_size;
                if (use_bias.HasValue && use_bias.Value) {
                    this.bias = new Parameter(torch.zeros(out_channels));
                } else {
                    this.bias = null;
                    //base.register_parameter("bias", null);
                }

                weight = new Parameter(torch.zeros(out_channels, in_channels / groups, kernel_size, kernel_size));

                offset_conv = torch.nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, (kernel_size, kernel_size),
                    (stride, stride), (padding, padding), (dilation, dilation), bias: true);
                ResetParameters();
            }

            private void ResetParameters()
            {
                torch.nn.init.kaiming_uniform_(weight, Math.Sqrt(5));
                if (use_bias.HasValue) {
                    long fanin = torch.nn.init.CalculateFanInAndFanOut(weight).fanIn;
                    var bound = 1 / Math.Sqrt(fanin);
                    torch.nn.init.uniform_(bias, -bound, bound);
                }
            }
            //TODO: Implement with offset too ???
            public override Tensor forward(Tensor input)
            {
                var offset = offset_conv.forward(input);
                offset = offset.contiguous().view(new long[] { -1, 2, kernel_size, kernel_size });
                input = torch.nn.functional.conv2d(input, weight, bias, strides, padding, dilation, groups);
                return input;
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                this.bias?.Dispose();
                this.weight?.Dispose();
                this.offset_conv?.Dispose();
            }
        }
    }
}
