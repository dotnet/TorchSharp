// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

using static TorchSharp.TensorExtensionMethods;

namespace TorchSharp.torchvision
{
    internal class Adjustment : AffineGridBase
    {
        protected Tensor Blend(Tensor img1, Tensor img2, double ratio)
        {
            var bound = img1.IsIntegral() ? 255.0 : 1.0;
            return (img1 * ratio + img2 * (1.0 - ratio)).clamp(0, bound).to(img2.dtype);
        }

        protected (Tensor h, Tensor s, Tensor v) RGBtoHSV(Tensor img)
        {
            var RGB = img.unbind(-3);
            var r = RGB[0];
            var g = RGB[1];
            var b = RGB[2];

            var maxc = torch.max(img, dimension: -3).values;
            var minc = torch.min(img, dimension: -3).values;

            var eqc = maxc == minc;
            var cr = maxc - minc;
            var ones = torch.ones_like(maxc);

            var s = cr / torch.where(eqc, ones, maxc);
            var cr_divisor = torch.where(eqc, ones, cr);

            var rc = (maxc - r) / cr_divisor;
            var gc = (maxc - g) / cr_divisor;
            var bc = (maxc - b) / cr_divisor;

            var hr = (maxc == r) * (bc - gc);
            var hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc);
            var hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc);

            var h = (hr + hg + hb);
            h = torch.fmod((h / 6.0 + 1.0), 1.0);

            return (h, s, maxc);
        }

        protected Tensor HSVtoRGB(Tensor h, Tensor s, Tensor v)
        {
            var h6 = h * 6.0;
            var i = torch.floor(h6);
            var f = h6 - i;
            i = i.to(torch.int32) % 6;

            var p = torch.clamp((v * (1.0 - s)), 0.0, 1.0);
            var q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0);
            var t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0);

            var iunsq = i.unsqueeze(dimension: -3);
            var mask = iunsq == torch.arange(6, device: i.device).view(-1, 1, 1);

            var a1 = torch.stack(new Tensor[] { v, q, p, p, t, v }, dimension: -3);
            var a2 = torch.stack(new Tensor[] { t, v, v, q, p, p }, dimension: -3);
            var a3 = torch.stack(new Tensor[] { p, p, t, v, v, q }, dimension: -3);
            var a4 = torch.stack(new Tensor[] { a1, a2, a3 }, dimension: -4);

            return torch.einsum("...ijk,...xijk ->...xjk", mask.to(h.dtype), a4);
        }
    }
}
