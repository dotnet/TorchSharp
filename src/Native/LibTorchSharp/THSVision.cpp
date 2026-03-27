// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSVision.h"

#include <torch/nn/init.h>

// The image processing code was ported from the Python implmentation found in:
//
// https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional_tensor.py

void _rgb_to_hsv(at::Tensor& img, at::Tensor& h, at::Tensor& saturation, at::Tensor& value)
{
    auto RGB = img.unbind(-3);
    auto r = RGB[0];
    auto g = RGB[1];
    auto b = RGB[2];

    auto maxc = std::get<0>(img.max(-3));
    auto minc = std::get<0>(img.min(-3));

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

    h = (hr + hg + hb);
    h = torch::fmod((h / 6.0 + 1.0), 1.0);

    saturation = s;
    value = maxc;
}

void _hsv_to_rgb(at::Tensor& h, at::Tensor& s, at::Tensor& v, at::Tensor& img)
{
    auto h6 = h * 6.0f;
    auto i = torch::floor(h6);
    auto f = h6 - i;
    i = i.to(at::ScalarType::Int) % 6;

    auto p = torch::clamp((v * (1.0f - s)), 0.0, 1.0);
    auto q = torch::clamp((v * (1.0 - s * f)), 0.0, 1.0);
    auto t = torch::clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0);

    auto iunsq = i.unsqueeze(-3);

    auto mask = iunsq == torch::arange(6, at::TensorOptions(i.device())).view({ -1, 1, 1 });

    auto a1 = torch::stack({ v, q, p, p, t, v }, -3);
    auto a2 = torch::stack({ t, v, v, q, p, p }, -3);
    auto a3 = torch::stack({ p, p, t, v, v, q }, -3);
    auto a4 = torch::stack({ a1, a2, a3 }, -4);

    img = torch::einsum("...ijk,...xijk ->...xjk", { mask.to(h.dtype()), a4 });
}

Tensor THSVision_AdjustHue(const Tensor i, const double hue_factor)
{
    try {
        torch_last_err = 0;

        auto img = *i;

        auto orig_dtype = img.scalar_type();

        if (!torch::isFloatingType(orig_dtype)) {
            img = img.to(c10::ScalarType::Float) / 255.0;
        }

        at::Tensor h;
        at::Tensor s;
        at::Tensor v;

        _rgb_to_hsv(img, h, s, v);

        h = (h + hue_factor) % 1.0;

        at::Tensor img_hue_adj;

        _hsv_to_rgb(h, s, v, img_hue_adj);

        if (!torch::isFloatingType(orig_dtype)) {
            img_hue_adj = (img_hue_adj * 255.0).to(orig_dtype);
        }

        return ResultTensor(img_hue_adj);
    }
    catch (const c10::Error e) {
        torch_last_err = strdup(e.what());
    }
    catch (const std::runtime_error e) {
        torch_last_err = strdup(e.what());
    }

    return nullptr;
}

Tensor THSVision_GenerateAffineGrid(Tensor theta, const int64_t w, const int64_t h, const int64_t ow, const int64_t oh)
{
    try {
        torch_last_err = 0;

        auto d = 0.5;

        auto thetaOptions = at::TensorOptions().dtype(theta->dtype()).device(theta->device());
        auto thetaDevice = at::TensorOptions().device(theta->device());

        auto base_grid = torch::empty({ 1, oh, ow, 3 }, thetaOptions);

        auto x_grid = torch::linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, ow, thetaDevice);
        base_grid.index({ at::indexing::Ellipsis, 0 }).copy_(x_grid);

        auto y_grid = torch::linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, oh, thetaDevice).unsqueeze_(-1);
        base_grid.index({ at::indexing::Ellipsis, 1 }).copy_(y_grid);
        base_grid.index({ at::indexing::Ellipsis, 2 }).fill_(1);

        auto rescaled_theta = theta->transpose(1, 2) / torch::tensor({ 0.5f * w, 0.5f * h }, thetaOptions);
        auto output_grid = base_grid.view({ 1, oh * ow, 3 }).bmm(rescaled_theta);

        return ResultTensor(output_grid.view({ 1, oh, ow, 2 }));
    }
    catch (const c10::Error e) {
        torch_last_err = strdup(e.what());
    }
    catch (const std::runtime_error e) {
        torch_last_err = strdup(e.what());
    }

    return nullptr;
}

Tensor THSVision_ApplyGridTransform(Tensor i, Tensor g, const int8_t m, const float* fill, const int64_t fill_length)
{
    try {
        torch_last_err = 0;

        auto img = *i;
        auto grid = *g;

        auto imgOptions = at::TensorOptions().dtype(img.dtype()).device(img.device());

        if (img.size(0) > 1) {
            grid = grid.expand({ img.size(0), grid.size(1), grid.size(2), grid.size(3) });
        }

        if (fill != nullptr) {
            auto dummy = torch::ones({ img.size(0), 1, img.size(2), img.size(3) }, imgOptions);
            img = torch::cat({ img, dummy }, 1);
        }

        const torch::nn::functional::GridSampleFuncOptions::mode_t mode =
            (m == 0)
            ? (torch::nn::functional::GridSampleFuncOptions::mode_t)torch::kNearest
            : (torch::nn::functional::GridSampleFuncOptions::mode_t)torch::kBilinear;
        auto sampleOpts = torch::nn::functional::GridSampleFuncOptions().padding_mode(torch::kZeros).mode(mode);
        sampleOpts.align_corners(false); // Supress warning. Default=false for grid_sample since libtorch 1.3.0

        img = torch::nn::functional::grid_sample(img, grid, sampleOpts);


        if (fill != nullptr) {

            auto COLON = at::indexing::Slice();

            auto mask = img.index({ COLON, at::indexing::Slice(-1, c10::nullopt), COLON,  COLON });
            img = img.index({ COLON, at::indexing::Slice(c10::nullopt, -1), COLON, COLON });
            mask = mask.expand_as(img);

            auto fill_img = torch::tensor(at::ArrayRef<float>(fill, fill_length), imgOptions).view({ 1, fill_length, 1, 1 }).expand_as(img);

            if (m == 0) {
                mask = mask < 0.5;
                img = torch::where(mask, fill_img, img);
            }
            else {
                img = img * mask + (-mask + 1.0) * fill_img;
            }
        }

        return ResultTensor(img);
    }
    catch (const c10::Error e) {
        torch_last_err = strdup(e.what());
    }
    catch (const std::runtime_error e) {
        torch_last_err = strdup(e.what());
    }

    return nullptr;
}

Tensor THSVision_ScaleChannel(Tensor ic)
{
    try {
        torch_last_err = 0;
        auto img_chan = *ic;

        auto hist = img_chan.is_cuda()
            ? torch::histc(img_chan.to(at::ScalarType::Float), 256, 0, 255)
            : torch::bincount(img_chan.view(-1), {}, 256);

        auto nonzero_hist = hist.index({ hist != 0 });

        auto step = torch::div(nonzero_hist.index({ at::indexing::Slice(c10::nullopt, -1) }).sum(), 255, "floor");
        auto count = step.count_nonzero();

        if (count.item<int>() == 0)
        {
            return ResultTensor(img_chan);
        }

        auto lut = torch::div(torch::cumsum(hist, 0) + torch::div(step, 2, "floor"), step, "floor");

        auto padOptions = torch::nn::functional::PadFuncOptions({ 1, 0 });
        lut = torch::nn::functional::pad(lut, padOptions).index({ at::indexing::Slice(c10::nullopt, -1) }).clamp(0, 255);

        return ResultTensor(lut.index({ img_chan.to(c10::ScalarType::Long) }).to(c10::ScalarType::Byte));
    }
    catch (const c10::Error e) {
        torch_last_err = strdup(e.what());
    }
    catch (const std::runtime_error e) {
        torch_last_err = strdup(e.what());
    }

    return nullptr;
}

void THSVision_ComputeOutputSize(const float* matrix, const int64_t matrix_length, const int64_t w, const int64_t h, int32_t* first, int32_t* second)
{
    try {
        torch_last_err = 0;

        auto pts = torch::tensor({ -0.5f * w, -0.5f * h, 1.0f, -0.5f * w, 0.5f * h, 1.0f, 0.5f * w, 0.5f * h, 1.0f, 0.5f * w, -0.5f * h, 1.0f }).reshape({ 4,3 });
        auto theta = torch::tensor(c10::ArrayRef<float>(matrix, matrix_length), c10::TensorOptions().dtype(c10::ScalarType::Float)).view({ 2, 3 });
        auto new_pts = torch::matmul(pts, theta.t());

        auto min_vals = std::get<0>(new_pts.min(0));
        auto max_vals = std::get<0>(new_pts.max(0));

        min_vals += torch::tensor({ w * 0.5f, h * 0.5f });
        max_vals += torch::tensor({ w * 0.5f, h * 0.5f });

        float tol = 1e-4;
        auto cmax = torch::ceil((max_vals / tol).trunc_() * tol);
        auto cmin = torch::floor((min_vals / tol).trunc_() * tol);

        auto size = cmax - cmin;

        *first = size[0].item<int>();
        *second = size[1].item<int>();
    }
    catch (const c10::Error e) {
        torch_last_err = strdup(e.what());
    }
    catch (const std::runtime_error e) {
        torch_last_err = strdup(e.what());
    }
}

Tensor THSVision_PerspectiveGrid(const float* c, const int64_t c_length, const int64_t ow, const int64_t oh, const int8_t scalar_type, const int device_type, const int device_index)
{
    try {
        torch_last_err = 0;

        auto fullOptions = at::TensorOptions()
            .dtype(at::ScalarType(scalar_type))
            .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index));
        auto devOptions = at::TensorOptions()
            .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index));

        auto coeffs = c10::ArrayRef<float>(c, c_length);

        auto theta1 = torch::tensor({ coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5] }, fullOptions).view({ 1, 2, 3 });
        auto theta2 = torch::tensor({ coeffs[6], coeffs[7], 1.0f, coeffs[6], coeffs[7], 1.0f }, fullOptions).view({ 1, 2, 3 });

        auto d = 0.5f;
        auto base_grid = torch::empty({ 1, oh, ow, 3 }, fullOptions);
        auto x_grid = torch::linspace(d, ow * 1.0 + d - 1.0, ow, devOptions);
        base_grid.index({ at::indexing::Ellipsis, 0 }).copy_(x_grid);

        auto y_grid = torch::linspace(d, oh * 1.0 + d - 1.0, oh, devOptions).unsqueeze_(-1);
        base_grid.index({ at::indexing::Ellipsis, 1 }).copy_(y_grid);
        base_grid.index({ at::indexing::Ellipsis, 2 }).fill_(1);

        auto rescaled_theta1 = theta1.transpose(1, 2) / torch::tensor({ 0.5f * ow, 0.5f * oh }, fullOptions);

        auto output_grid1 = base_grid.view({ 1, oh * ow, 3 }).bmm(rescaled_theta1);
        auto output_grid2 = base_grid.view({ 1, oh * ow, 3 }).bmm(theta2.transpose(1, 2));
        auto output_grid = output_grid1 / output_grid2 - 1.0f;

        return ResultTensor(output_grid.view({ 1, oh, ow, 2 }));
    }
    catch (const c10::Error e) {
        torch_last_err = strdup(e.what());
    }
    catch (const std::runtime_error e) {
        torch_last_err = strdup(e.what());
    }

    return nullptr;
}


void THSVision_BRGA_RGB(const uint8_t* inputBytes, uint8_t* redBytes, uint8_t* greenBytes, uint8_t* blueBytes, int64_t inputChannelCount, int64_t imageSize)
{
    const int inputBlue = 0, inputGreen = 1, inputRed = 2;

    for (int64_t i = 0, j = 0; i < imageSize; i += 1, j += inputChannelCount) {
        redBytes[i] = inputBytes[inputRed + j];
        greenBytes[i] = inputBytes[inputGreen + j];
        blueBytes[i] = inputBytes[inputBlue + j];
    }
}

void THSVision_BRGA_RGBA(const uint8_t* inputBytes, uint8_t* redBytes, uint8_t* greenBytes, uint8_t* blueBytes, uint8_t* alphaBytes, int64_t inputChannelCount, int64_t imageSize)
{
    const int inputBlue = 0, inputGreen = 1, inputRed = 2, inputAlpha = 3;

    bool inputHasAlpha = inputChannelCount == 4;

    for (int64_t i = 0, j = 0; i < imageSize; i += 1, j += inputChannelCount) {
        redBytes[i] = inputBytes[inputRed + j];
        greenBytes[i] = inputBytes[inputGreen + j];
        blueBytes[i] = inputBytes[inputBlue + j];
        alphaBytes[i] = inputHasAlpha ? inputBytes[inputAlpha + j] : 255;
    }
}

void THSVision_RGB_BRGA(const uint8_t* inputBytes, uint8_t* outBytes, int64_t inputChannelCount, int64_t imageSize)
{
    bool isgrey = inputChannelCount == 1;
    bool inputHasAlpha = inputChannelCount == 4;

    const int inputRed = 0, inputGreen = imageSize, inputBlue = imageSize * 2, inputAlpha = imageSize * 3;
    const int outputBlue = 0, outputGreen = 1, outputRed = 2, outputAlpha = 3;

    for (int64_t i = 0, j = 0; i < imageSize; i += 1, j += 4) {
        auto redPixel = inputBytes[inputRed + i];
        outBytes[outputRed + j] = redPixel;
        if (!isgrey) {
            outBytes[outputGreen + j] = inputBytes[inputGreen + i];
            outBytes[outputBlue + j] = inputBytes[inputBlue + i];
        }
        else {
            outBytes[outputGreen + j] = redPixel;
            outBytes[outputBlue + j] = redPixel;
        }
        outBytes[outputAlpha + j] = inputHasAlpha ? inputBytes[inputBlue + i] : 255;
    }
}