// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSVision.h"

#include <torch/nn/init.h>

Tensor THSVision_RGBtoHSV(const Tensor img, Tensor* saturation, Tensor* value)
{
    CATCH(
        auto RGB = img->unbind(-3);
        auto r = RGB[0];
        auto g = RGB[1];
        auto b = RGB[2];

        auto maxc = std::get<0>(img->max(-3));
        auto minc = std::get<0>(img->min(-3));

        auto eqc = maxc == minc;
        auto cr = maxc - minc;

        auto options = at::TensorOptions()
            .dtype(maxc.dtype())
            .device(maxc.device())
            .requires_grad(maxc.requires_grad());

        auto ones = torch::ones_like(maxc, options);

        auto s = cr / torch::where(eqc, ones, maxc);
        auto cr_divisor = torch::where(eqc, ones, cr);

        auto rc = (maxc - r) / cr_divisor;
        auto gc = (maxc - g) / cr_divisor;
        auto bc = (maxc - b) / cr_divisor;

        auto hr = (maxc == r) * (bc - gc);
        auto hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc);
        auto hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc);

        auto h = (hr + hg + hb);
        h = torch::fmod((h / 6.0 + 1.0), 1.0);

        *saturation = ResultTensor(s);
        *value = ResultTensor(maxc);
        return ResultTensor(h);
    );

    return nullptr;
}

Tensor THSVision_HSVtoRGB(const Tensor h, const Tensor s, const Tensor v)
{
    CATCH(
        auto h6 = (*h) * 6.0f;
        auto i = torch::floor(h6);
        auto f = h6 - i;
        i = i.to(at::ScalarType::Int) % 6;

        auto p = torch::clamp((*v * (1.0f - *s)), 0.0, 1.0);
        auto q = torch::clamp((*v * (1.0 - *s * f)), 0.0, 1.0);
        auto t = torch::clamp((*v * (1.0 - *s * (1.0 - f))), 0.0, 1.0);

        auto iunsq = i.unsqueeze(-3);

        auto mask = iunsq == torch::arange(6, at::TensorOptions(i.device())).view({ -1, 1, 1 });

        auto a1 = torch::stack({ *v, q, p, p, t, *v }, -3);
        auto a2 = torch::stack({ t, *v, *v, q, p, p }, -3);
        auto a3 = torch::stack({ p, p, t, *v, *v, q }, -3);
        auto a4 = torch::stack({ a1, a2, a3 }, -4);

        auto img = torch::einsum("...ijk,...xijk ->...xjk", { mask.to(h->dtype()), a4 });

        return ResultTensor(img);
    );

    return nullptr;
}
