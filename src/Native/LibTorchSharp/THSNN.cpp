// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSNN.h"

#include <torch/nn/init.h>


NNModule THSNN_Identity_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        res = create_module<torch::nn::IdentityImpl>(outAsAnyModule);
    );
}

Tensor THSNN_Identity_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Identity>()->forward(*tensor));
}

NNModule THSNN_Linear_ctor(const int64_t input_size, const int64_t output_size, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::LinearOptions(input_size, output_size).bias(bias);
        res = create_module<torch::nn::LinearImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Linear_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Linear>()->forward(*tensor));
}

Tensor THSNN_Linear_bias(const NNModule module)
{
    return get_bias<torch::nn::Linear>(module);
}

void THSNN_Linear_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::Linear>(module, bias);
}

Tensor THSNN_Linear_weight(const NNModule module)
{
    return get_weight<torch::nn::Linear>(module);
}

void THSNN_Linear_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::Linear>(module, weight);
}

Tensor THSNN_functional_linear(const Tensor input, const Tensor weights, const Tensor bias)
{
    CATCH_TENSOR(bias == nullptr ?
        torch::nn::functional::linear(*input, *weights) :
        torch::nn::functional::linear(*input, *weights, *bias));
}

Tensor THSNN_functional_bilinear(const Tensor input1, const Tensor input2, const Tensor weights, const Tensor bias)
{
    CATCH_TENSOR(bias == nullptr ?
        torch::nn::functional::bilinear(*input1, *input2, *weights) :
        torch::nn::functional::bilinear(*input1, *input2, *weights, *bias));
}

NNModule THSNN_Bilinear_ctor(const int64_t input_size_1, const int64_t input_size_2, const int64_t output_size, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::BilinearOptions(input_size_1, input_size_2, output_size).bias(bias);
        res = create_module<torch::nn::BilinearImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Bilinear_forward(const NNModule module, const Tensor x1, const Tensor x2)
{
    CATCH_TENSOR((*module)->as<torch::nn::Bilinear>()->forward(*x1, *x2));
}

Tensor THSNN_Bilinear_bias(const NNModule module)
{
    return get_bias<torch::nn::Bilinear>(module);
}

void THSNN_Bilinear_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::Bilinear>(module, bias);
}

Tensor THSNN_Bilinear_weight(const NNModule module)
{
    return get_weight<torch::nn::Bilinear>(module);
}

void THSNN_Bilinear_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::Bilinear>(module, weight);
}

Tensor THSNN_dropout(const Tensor input, const double p, bool training, bool inplace)
{
    auto opts = torch::nn::functional::DropoutFuncOptions()
        .inplace(inplace)
        .training(training)
        .p(p);

    CATCH_TENSOR(torch::nn::functional::dropout(*input, opts));
}

Tensor THSNN_dropout2d(const Tensor input, const double p, bool training, bool inplace)
{
    auto opts = torch::nn::functional::Dropout2dFuncOptions()
        .inplace(inplace)
        .training(training)
        .p(p);

    CATCH_TENSOR(torch::nn::functional::dropout2d(*input, opts));
}

Tensor THSNN_dropout3d(const Tensor input, const double p, bool training, bool inplace)
{
    auto opts = torch::nn::functional::Dropout3dFuncOptions()
        .inplace(inplace)
        .training(training)
        .p(p);

    CATCH_TENSOR(torch::nn::functional::dropout3d(*input, opts));
}

Tensor THSNN_alpha_dropout(const Tensor input, const double p, bool training, bool inplace)
{
    auto opts = torch::nn::functional::AlphaDropoutFuncOptions()
        .inplace(inplace)
        .training(training)
        .p(p);

    CATCH_TENSOR(torch::nn::functional::alpha_dropout(*input, opts));
}

Tensor THSNN_feature_alpha_dropout(const Tensor input, const double p, bool training, bool inplace)
{
    auto opts = torch::nn::functional::FeatureAlphaDropoutFuncOptions()
        .inplace(inplace)
        .training(training)
        .p(p);

    CATCH_TENSOR(torch::nn::functional::feature_alpha_dropout(*input, opts));
}

NNModule THSNN_Dropout_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::DropoutOptions(probability).inplace(inplace);
        res = create_module<torch::nn::DropoutImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Dropout_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Dropout>()->forward(*tensor));
}

NNModule THSNN_AlphaDropout_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AlphaDropoutOptions(probability).inplace(inplace);
    res = create_module<torch::nn::AlphaDropoutImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AlphaDropout_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AlphaDropout>()->forward(*tensor));
}

NNModule THSNN_Dropout1d_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        // Creating a Dropout2d instance here is done on purpose. There's no torch::nn::Dropout1d
        auto opts = torch::nn::Dropout2dOptions(probability).inplace(inplace);
        res = create_module<torch::nn::Dropout2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Dropout1d_forward(const NNModule module, const Tensor tensor)
{
    auto drop1d = (*module)->as<torch::nn::Dropout2d>();
    CATCH_TENSOR(drop1d->options.inplace()
        ? drop1d->forward((*tensor).unsqueeze_(-1)).squeeze_(-1)
        : drop1d->forward((*tensor).unsqueeze(-1)).squeeze(-1));
}


NNModule THSNN_Dropout2d_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Dropout2dOptions(probability).inplace(inplace);
        res = create_module<torch::nn::Dropout2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Dropout2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Dropout2d>()->forward(*tensor));
}

NNModule THSNN_Dropout3d_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Dropout3dOptions(probability).inplace(inplace);
        res = create_module<torch::nn::Dropout3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Dropout3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Dropout3d>()->forward(*tensor));
}

NNModule THSNN_FeatureAlphaDropout_ctor(double probability, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::FeatureAlphaDropoutOptions(probability);
        res = create_module<torch::nn::FeatureAlphaDropoutImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_FeatureAlphaDropout_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::FeatureAlphaDropout>()->forward(*tensor));
}

NNModule THSNN_PixelShuffle_ctor(const int64_t upscale_factor, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::PixelShuffleOptions(upscale_factor);
        res = create_module<torch::nn::PixelShuffleImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_PixelShuffle_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::PixelShuffle>()->forward(*tensor));
}

NNModule THSNN_PixelUnshuffle_ctor(const int64_t downscale_factor, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::PixelUnshuffleOptions(downscale_factor);
        res = create_module<torch::nn::PixelUnshuffleImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_PixelUnshuffle_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::PixelUnshuffle>()->forward(*tensor));
}

template<typename T>
void ApplyUpsampleMode(T& opts, const int8_t mode)
{
    if (mode == 0)
        opts = opts.mode(torch::kNearest);
    if (mode == 1)
        opts = opts.mode(torch::kLinear);
    if (mode == 2)
        opts = opts.mode(torch::kBilinear);
    if (mode == 3)
        opts = opts.mode(torch::kBicubic);
    if (mode == 4)
        opts = opts.mode(torch::kTrilinear);
}

template<typename T>
void ApplyInterpolateMode(T& opts, const int8_t mode)
{
    if (mode == 0)
        opts = opts.mode(torch::kNearest);
    if (mode == 1)
        opts = opts.mode(torch::kLinear);
    if (mode == 2)
        opts = opts.mode(torch::kBilinear);
    if (mode == 3)
        opts = opts.mode(torch::kBicubic);
    if (mode == 4)
        opts = opts.mode(torch::kTrilinear);
    if (mode == 5)
        opts = opts.mode(torch::kArea);
}

NNModule THSNN_Upsample_ctor(const int64_t* size, const int size_len, const double* scale_factor, const int scale_factor_len, const int8_t mode, const int8_t align_corners, NNAnyModule* outAsAnyModule)
{
    auto opts = torch::nn::UpsampleOptions();
    // align_corners -- 0=None, 1=true, 2=false
    if (align_corners != 0)
        opts.align_corners(align_corners == 1);
    ApplyUpsampleMode(opts, mode);

    CATCH_RETURN_NNModule(
        if (size_len > 0) {
            std::vector<int64_t> sizes;
            for (int i = 0; i < size_len; ++i) {
                sizes.push_back(size[i]);
            }
            opts.size(sizes);
        }
        if (scale_factor_len > 0) {
            std::vector<double> scales;
            for (int i = 0; i < scale_factor_len; ++i) {
                scales.push_back(scale_factor[i]);
            }
            opts.scale_factor(scales);
        }
        res = create_module<torch::nn::UpsampleImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_Upsample_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Upsample>()->forward(*tensor));
}

template<typename T>
void ApplyPadMode(T& opts, const int64_t padding)
{
    if (padding == 1)
        opts = opts.mode(torch::kReflect);
    if (padding == 2)
        opts = opts.mode(torch::kReplicate);
    if (padding == 3)
        opts = opts.mode(torch::kCircular);
    if (padding == 4)
        opts = opts.mode(torch::kConstant);
}

template<typename T>
void ApplyGridMode(T& opts, const int8_t mode)
{
    if (mode == 0)
        opts = opts.mode(torch::kNearest);
    if (mode == 2)
        opts = opts.mode(torch::kBilinear);
    // The PyTorch docs say that bicubic should be supported, but the C++
    // mode type does not allow for it. I'm leaving this in for future use.
    //if (mode == 3)
    //    opts = opts.mode(torch::kBicubic);
}

template<typename T>
void ApplyGridPadMode(T& opts, const int64_t padding)
{
    if (padding == 0)
        opts = opts.padding_mode(torch::kZeros);
    if (padding == 1)
        opts = opts.padding_mode(torch::kReflection);
    if (padding == 2)
        opts = opts.padding_mode(torch::kBorder);
}

Tensor THSNN_pad(const Tensor input, const int64_t* pad, const int pad_length, const int8_t mode, const double value)
{
    std::vector<int64_t> padding;
    for (int i = 0; i < pad_length; ++i) {
        padding.push_back(pad[i]);
    }
    auto opts = torch::nn::functional::PadFuncOptions(padding).value(value);
    ApplyPadMode(opts, mode);

    CATCH_TENSOR(torch::nn::functional::pad(*input, opts));
}

Tensor THSNN_grid_sample(const Tensor input, const Tensor grid, const int8_t mode, const int8_t padding_mode, const int8_t align_corners)
{
    auto opts = torch::nn::functional::GridSampleFuncOptions();
    if (align_corners != 0)
        opts.align_corners(align_corners == 1);
    ApplyGridMode(opts, mode);
    ApplyGridPadMode(opts, padding_mode);
    CATCH_TENSOR(torch::nn::functional::grid_sample(*input, *grid, opts));
}

Tensor THSNN_affine_grid(const Tensor theta, const int64_t* size, const int size_len, const bool align_corners)
{
    CATCH_TENSOR(torch::nn::functional::affine_grid(*theta, at::ArrayRef<int64_t>(size, size_len), align_corners));
}


EXPORT_API(Tensor) THSNN_interpolate(const Tensor input, const int64_t* size, const int size_len, const double* scale_factor, const int scale_factor_len, const int8_t mode, const int8_t align_corners, const bool recompute_scale_factor, NNAnyModule* outAsAnyModule)
{
    auto opts = torch::nn::functional::InterpolateFuncOptions().recompute_scale_factor(recompute_scale_factor);
    // align_corners -- 0=None, 1=true, 2=false
    if (align_corners != 0)
        opts.align_corners(align_corners == 1);
    ApplyInterpolateMode(opts, mode);

    if (size_len > 0) {
        std::vector<int64_t> sizes;
        for (int i = 0; i < size_len; ++i) {
            sizes.push_back(size[i]);
        }
        opts.size(sizes);
    }
    if (scale_factor_len > 0) {
        std::vector<double> scales;
        for (int i = 0; i < scale_factor_len; ++i) {
            scales.push_back(scale_factor[i]);
        }
        opts.scale_factor(scales);
    }

    CATCH_TENSOR(torch::nn::functional::interpolate(*input, opts));
}

NNModule THSNN_Embedding_ctor(const int64_t num_embeddings, const int64_t embedding_dims,
    const int64_t padding_idx, bool has_pi, const double max_norm, const bool has_mn, const double norm_type,
    const bool scale_grad_by_freq, const bool sparse,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::EmbeddingOptions(num_embeddings, embedding_dims)
            .norm_type(norm_type)
            .scale_grad_by_freq(scale_grad_by_freq)
            .sparse(sparse);

        if (has_pi)
            opts.padding_idx(padding_idx);
        if (has_mn)
            opts.max_norm(max_norm);

        res = create_module<torch::nn::EmbeddingImpl>(opts, outAsAnyModule);
    );
}

NNModule THSNN_Embedding_from_pretrained(const Tensor embeddings, const bool freeze,
    const int64_t padding_idx, bool has_pi, const double max_norm, const bool has_mn, const double norm_type,
    const bool scale_grad_by_freq, const bool sparse,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto rows = embeddings->size(0);
        auto cols = embeddings->size(1);

        auto opts = torch::nn::EmbeddingOptions(rows, cols)
            .norm_type(norm_type)
            .scale_grad_by_freq(scale_grad_by_freq)
            .sparse(sparse);

        if (has_pi)
            opts.padding_idx(padding_idx);
        if (has_mn)
            opts.max_norm(max_norm);

        // Can't use the template function here -- custom logic.
        auto mod = std::make_shared<torch::nn::EmbeddingImpl>(opts);
        mod->weight = *embeddings;
        mod->weight.set_requires_grad(!freeze);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::EmbeddingImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }
        res = new std::shared_ptr<torch::nn::Module>(mod);
    );
}

Tensor THSNN_Embedding_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Embedding>()->forward(*tensor));
}

Tensor THSNN_Embedding_weight(const NNModule module)
{
    return get_weight<torch::nn::Embedding>(module);
}

void THSNN_Embedding_set_weight(const NNModule module, const Tensor weights)
{
    set_weight<torch::nn::Embedding>(module, weights);
}

template<typename T>
void ApplyEmbeddingBagMode(T& opts, const int64_t mode)
{
    if (mode == 0)
        opts = opts.mode(torch::kSum);
    if (mode == 1)
        opts = opts.mode(torch::kMean);
    if (mode == 2)
        opts = opts.mode(torch::kMax);
}

NNModule THSNN_EmbeddingBag_ctor(const int64_t num_embeddings, const int64_t embedding_dims,
    const double max_norm, const bool has_mn, const double norm_type, const bool scale_grad_by_freq,
    const int64_t mode, const bool sparse, const bool include_last_offset, const int64_t padding_idx,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::EmbeddingBagOptions(num_embeddings, embedding_dims)
            .norm_type(norm_type)
            .scale_grad_by_freq(scale_grad_by_freq)
            .include_last_offset(include_last_offset)
            .sparse(sparse);

        ApplyEmbeddingBagMode(opts, mode);

        if (has_mn)
            opts = opts.max_norm(max_norm);
        if (padding_idx >= 0)
            opts = opts.padding_idx(padding_idx);

        res = create_module<torch::nn::EmbeddingBagImpl>(opts, outAsAnyModule);
    );
}

NNModule THSNN_EmbeddingBag_from_pretrained(const Tensor embeddings, const bool freeze,
    const double max_norm, const bool has_mn, const double norm_type, const bool scale_grad_by_freq,
    const int64_t mode, const bool sparse, const bool include_last_offset, const int64_t padding_idx,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto rows = embeddings->size(0);
        auto cols = embeddings->size(1);

        auto opts = torch::nn::EmbeddingBagOptions(rows, cols)
            .norm_type(norm_type)
            .scale_grad_by_freq(scale_grad_by_freq)
            .include_last_offset(include_last_offset)
            .sparse(sparse);

        ApplyEmbeddingBagMode(opts, mode);

        if (has_mn)
            opts.max_norm(max_norm);
        if (padding_idx >= 0)
            opts = opts.padding_idx(padding_idx);

        // Can't use the template function here -- custom logic.
        auto mod = std::make_shared<torch::nn::EmbeddingBagImpl>(opts);
        mod->weight = *embeddings;
        mod->weight.set_requires_grad(!freeze);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::EmbeddingBagImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }
        res = new std::shared_ptr<torch::nn::Module>(mod);
    );
}

Tensor THSNN_EmbeddingBag_forward(const NNModule module, const Tensor input, const Tensor offsets, const Tensor per_sample_weights)
{
    if (offsets != nullptr && per_sample_weights != nullptr)
    {
        CATCH_TENSOR((*module)->as<torch::nn::EmbeddingBag>()->forward(*input, *offsets, *per_sample_weights));
    }
    else if (offsets == nullptr && per_sample_weights != nullptr)
    {
        CATCH_TENSOR((*module)->as<torch::nn::EmbeddingBag>()->forward(*input, {}, *per_sample_weights));
    }
    else if (offsets != nullptr && per_sample_weights == nullptr)
    {
        CATCH_TENSOR((*module)->as<torch::nn::EmbeddingBag>()->forward(*input, *offsets));
    }
    else
    {
        CATCH_TENSOR((*module)->as<torch::nn::EmbeddingBag>()->forward(*input));
    }
}

Tensor THSNN_EmbeddingBag_weight(const NNModule module)
{
    return get_weight<torch::nn::EmbeddingBag>(module);
}

void THSNN_EmbeddingBag_set_weight(const NNModule module, const Tensor weights)
{
    set_weight<torch::nn::EmbeddingBag>(module, weights);
}

template<typename T>
void ApplyTransformerActivation(T& opts, const int64_t activation)
{
    if (activation == 0)
        opts = opts.activation(torch::kReLU);
    if (activation == 1)
        opts = opts.activation(torch::kGELU);
}

template<typename T>
void ApplyRnnActivation(T& opts, const int64_t activation)
{
    if (activation == 0)
        opts = opts.nonlinearity(torch::kReLU);
    if (activation == 1)
        opts = opts.nonlinearity(torch::kTanh);
}

NNModule THSNN_Transformer_ctor(const int64_t d_model, const int64_t nhead, const int64_t num_encoder_layers, const int64_t num_decoder_layers, const int64_t dim_feedforward, const double dropout, const int64_t activation, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::TransformerOptions(d_model, nhead)
            .num_encoder_layers(num_encoder_layers)
            .num_decoder_layers(num_decoder_layers)
            .dim_feedforward(dim_feedforward)
            .dropout(dropout);
        ApplyTransformerActivation(opts, activation);

        res = create_module<torch::nn::TransformerImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_Transformer_forward(const NNModule module, const Tensor src, const Tensor tgt, const Tensor src_mask, const Tensor tgt_mask, const Tensor memory_mask, const Tensor src_key_padding_mask, const Tensor tgt_key_padding_mask, const Tensor memory_key_padding_mask)
{
    CATCH_TENSOR((*module)->as<torch::nn::Transformer>()->forward(
        *src,
        *tgt,
        (src_mask ? *src_mask : at::Tensor()),
        (tgt_mask ? *tgt_mask : at::Tensor()),
        (memory_mask ? *memory_mask : at::Tensor()),
        (src_key_padding_mask ? *src_key_padding_mask : at::Tensor()),
        (tgt_key_padding_mask ? *tgt_key_padding_mask : at::Tensor()),
        (memory_key_padding_mask ? *memory_key_padding_mask : at::Tensor()))
    );
}

NNModule THSNN_TransformerEncoderLayer_ctor(const int64_t d_model, const int64_t nhead, const int64_t dim_feedforward, const double dropout, const int64_t activation, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
            .dim_feedforward(dim_feedforward)
            .dropout(dropout);
        ApplyTransformerActivation(opts, activation);

        res = create_module<torch::nn::TransformerEncoderLayerImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_TransformerEncoderLayer_forward(const NNModule module, const Tensor src, const Tensor src_mask, const Tensor src_key_padding_mask)
{
    CATCH_TENSOR((*module)->as<torch::nn::TransformerEncoderLayer>()->forward(
        *src,
        (src_mask ? *src_mask : at::Tensor()),
        (src_key_padding_mask ? *src_key_padding_mask : at::Tensor()))
    );
}

NNModule THSNN_TransformerDecoderLayer_ctor(const int64_t d_model, const int64_t nhead, const int64_t dim_feedforward, const double dropout, const int64_t activation, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::TransformerDecoderLayerOptions(d_model, nhead)
            .dim_feedforward(dim_feedforward)
            .dropout(dropout);
        ApplyTransformerActivation(opts, activation);

        res = create_module<torch::nn::TransformerDecoderLayerImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_TransformerDecoderLayer_forward(const NNModule module, const Tensor tgt, const Tensor memory, const Tensor tgt_mask, const Tensor memory_mask, const Tensor tgt_key_padding_mask, const Tensor memory_key_padding_mask)
{
    CATCH_TENSOR((*module)->as<torch::nn::TransformerDecoderLayer>()->forward(
        *tgt,
        *memory,
        (tgt_mask ? *tgt_mask : at::Tensor()),
        (memory_mask ? *memory_mask : at::Tensor()),
        (tgt_key_padding_mask ? *tgt_key_padding_mask : at::Tensor()),
        (memory_key_padding_mask ? *memory_key_padding_mask : at::Tensor()))
    );
}

NNModule THSNN_MultiheadAttention_ctor(const int64_t embeded_dim, const int64_t num_heads, const double dropout, const bool bias, const bool add_bias_kv, const bool add_zero_attn, const int64_t kdim, const int64_t vdim, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::MultiheadAttentionOptions(embeded_dim, num_heads)
        .dropout(dropout)
        .bias(bias)
        .add_bias_kv(add_bias_kv)
        .add_zero_attn(add_zero_attn)
        .kdim(kdim)
        .vdim(vdim);

    res = create_module<torch::nn::MultiheadAttentionImpl>(opts, outAsAnyModule);
    );
}

void THSNN_MultiheadAttention_forward(const NNModule module, const Tensor query, const Tensor key, const Tensor value, const Tensor key_padding_mask, const bool need_weights, const Tensor attn_mask, Tensor& res1, Tensor& res2)
{
    CATCH_TENSORS_2((*module)->as<torch::nn::MultiheadAttention>()->forward(
        *query,
        *key,
        *value,
        (key_padding_mask ? *key_padding_mask : at::Tensor()),
        need_weights,
        (attn_mask ? *attn_mask : at::Tensor()))
    );
}

NNModule THSNN_TransformerEncoder_ctor(const NNModule encoder_layer, const int64_t num_layers, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto enc = (*encoder_layer)->as<torch::nn::TransformerEncoderLayer>();
        auto opts = torch::nn::TransformerEncoderOptions(torch::nn::TransformerEncoderLayer(*enc), num_layers);

        res = create_module<torch::nn::TransformerEncoderImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_TransformerEncoder_forward(const NNModule module, const Tensor src, const Tensor src_mask, const Tensor src_key_padding_mask)
{
    CATCH_TENSOR((*module)->as<torch::nn::TransformerEncoder>()->forward(
        *src,
        (src_mask ? *src_mask : at::Tensor()),
        (src_key_padding_mask ? *src_key_padding_mask : at::Tensor()))
    );
}

NNModule THSNN_TransformerDecoder_ctor(const NNModule decoder_layer, const int64_t num_layers, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto dec = (*decoder_layer)->as<torch::nn::TransformerDecoderLayer>();
        auto opts = torch::nn::TransformerDecoderOptions(torch::nn::TransformerDecoderLayer(*dec), num_layers);

        res = create_module<torch::nn::TransformerDecoderImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_TransformerDecoder_forward(const NNModule module, const Tensor tgt, const Tensor memory, const Tensor tgt_mask, const Tensor memory_mask, const Tensor tgt_key_padding_mask, const Tensor memory_key_padding_mask)
{
    CATCH_TENSOR((*module)->as<torch::nn::TransformerDecoder>()->forward(
        *tgt,
        *memory,
        (tgt_mask ? *tgt_mask : at::Tensor()),
        (memory_mask ? *memory_mask : at::Tensor()),
        (tgt_key_padding_mask ? *tgt_key_padding_mask : at::Tensor()),
        (memory_key_padding_mask ? *memory_key_padding_mask : at::Tensor()))
    );
}

NNModule THSNN_Flatten_ctor(const int64_t start_dim, const int64_t end_dim, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::FlattenOptions()
            .start_dim(start_dim)
            .end_dim(end_dim);

        res = create_module<torch::nn::FlattenImpl>(opts, outAsAnyModule);
    );
}
Tensor   THSNN_Flatten_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Flatten>()->forward(*tensor));
}

NNModule THSNN_Unflatten_ctor(const int64_t dim, const int64_t* shape, const int64_t shape_len, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        std::vector<int64_t> sizes;
        for (int64_t i = 0; i < shape_len; ++i)
        {
            sizes.push_back(shape[i]);
        }
        auto opts = torch::nn::UnflattenOptions(dim, sizes);
        res = create_module<torch::nn::UnflattenImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Unflatten_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Unflatten>()->forward(*tensor));
}

NNModule THSNN_CosineSimilarity_ctor(const int64_t dim, double eps, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::CosineSimilarityOptions()
        .dim(dim)
        .eps(eps);

        res = create_module<torch::nn::CosineSimilarityImpl>(opts, outAsAnyModule);
    );

}

Tensor  THSNN_CosineSimilarity_forward(const NNModule module, const Tensor input1, const Tensor input2)
{
    CATCH_TENSOR((*module)->as<torch::nn::CosineSimilarity>()->forward(*input1, *input2));
}

NNModule THSNN_PairwiseDistance_ctor(double p, double eps, bool keep_dim, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::PairwiseDistanceOptions()
        .p(p)
        .eps(eps)
        .keepdim(keep_dim);

        res = create_module<torch::nn::PairwiseDistanceImpl>(opts, outAsAnyModule);
    );

}

Tensor  THSNN_PairwiseDistance_forward(const NNModule module, const Tensor input1, const Tensor input2)
{
    CATCH_TENSOR((*module)->as<torch::nn::PairwiseDistance>()->forward(*input1, *input2));
}

NNModule THSNN_RNN_ctor(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const int64_t nonlinearity, const bool bias, const bool batchFirst, const double dropout, const bool bidirectional, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::RNNOptions(input_size, hidden_size)
            .num_layers(num_layers)
            .bias(bias)
            .batch_first(batchFirst)
            .dropout(dropout)
            .bidirectional(bidirectional);

        ApplyRnnActivation(opts, nonlinearity);

        res = create_module<torch::nn::RNNImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_RNN_forward(const NNModule module, const Tensor input1, const Tensor input2, Tensor* h_n)
{
    Tensor output = nullptr;
    *h_n = nullptr;
    CATCH(
        auto result = (*module)->as<torch::nn::RNN>()->forward(*input1, (input2 ? *input2 : at::Tensor()));
        output = new torch::Tensor(std::get<0>(result));
        *h_n = new torch::Tensor(std::get<1>(result));
    );
    return output;
}

PackedSequence THSNN_RNN_forward_with_packed_input(const NNModule module, const PackedSequence input1, const Tensor input2, Tensor* h_n)
{
    PackedSequence output = nullptr;
    *h_n = nullptr;
    CATCH(
        auto result = (*module)->as<torch::nn::RNN>()->forward_with_packed_input(*input1, (input2 ? *input2 : at::Tensor()));
        output = new torch::nn::utils::rnn::PackedSequence(std::get<0>(result));
        *h_n = new torch::Tensor(std::get<1>(result));
    );
    return output;
}

void THSNN_RNN_flatten_parameters(const NNModule module)
{
    CATCH(
        (*module)->as<torch::nn::RNN>()->flatten_parameters();
    );
}

Tensor THSNN_RNN_bias_ih(const NNModule module, const int64_t idx)
{
    return get_bias_ih<torch::nn::RNN>(module, idx);
}

void THSNN_RNN_set_bias_ih(const NNModule module, const Tensor bias, const int64_t idx)
{
    set_bias_ih<torch::nn::RNN>(module, bias, idx);
}

Tensor THSNN_RNN_weight_ih(const NNModule module, const int64_t idx)
{
    return get_weight_ih<torch::nn::RNN>(module, idx);
}

void THSNN_RNN_set_weight_ih(const NNModule module, const Tensor weight, const int64_t idx)
{
    set_weight_ih<torch::nn::RNN>(module, weight, idx);
}

Tensor THSNN_RNN_bias_hh(const NNModule module, const int64_t idx)
{
    return get_bias_hh<torch::nn::RNN>(module, idx);
}

void THSNN_RNN_set_bias_hh(const NNModule module, const Tensor bias, const int64_t idx)
{
    set_bias_hh<torch::nn::RNN>(module, bias, idx);
}

Tensor THSNN_RNN_weight_hh(const NNModule module, const int64_t idx)
{
    return get_weight_hh<torch::nn::RNN>(module, idx);
}

void THSNN_RNN_set_weight_hh(const NNModule module, const Tensor weight, const int64_t idx)
{
    set_weight_hh<torch::nn::RNN>(module, weight, idx);
}


NNModule THSNN_GRU_ctor(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const bool bias, const bool batchFirst, const double dropout, const bool bidirectional, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::GRUOptions(input_size, hidden_size)
            .num_layers(num_layers)
            .bias(bias)
            .batch_first(batchFirst)
            .dropout(dropout)
            .bidirectional(bidirectional);

        res = create_module<torch::nn::GRUImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_GRU_forward(const NNModule module, const Tensor input1, const Tensor input2, Tensor* h_n)
{
    Tensor output = nullptr;
    *h_n = nullptr;
    CATCH(
        auto result = (*module)->as<torch::nn::GRU>()->forward(*input1, (input2 ? *input2 : at::Tensor()));
        output = new torch::Tensor(std::get<0>(result));
        *h_n = new torch::Tensor(std::get<1>(result));
    );
    return output;
}

PackedSequence THSNN_GRU_forward_with_packed_input(const NNModule module, const PackedSequence input1, const Tensor input2, Tensor* h_n)
{
    PackedSequence output = nullptr;
    *h_n = nullptr;
    CATCH(
        auto result = (*module)->as<torch::nn::GRU>()->forward_with_packed_input(*input1, (input2 ? *input2 : at::Tensor()));
        output = new torch::nn::utils::rnn::PackedSequence(std::get<0>(result));
        *h_n = new torch::Tensor(std::get<1>(result));
    );
    return output;
}

void THSNN_GRU_flatten_parameters(const NNModule module)
{
    CATCH(
        (*module)->as<torch::nn::GRU>()->flatten_parameters();
    );
}

NNModule THSNN_LSTM_ctor(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const bool bias, const bool batchFirst, const double dropout, const bool bidirectional, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::LSTMOptions(input_size, hidden_size)
        .num_layers(num_layers)
        .bias(bias)
        .batch_first(batchFirst)
        .dropout(dropout)
        .bidirectional(bidirectional);

        res = create_module<torch::nn::LSTMImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_LSTM_forward(const NNModule module, const Tensor input1, const Tensor h0, const Tensor c0, Tensor* h_n, Tensor* c_n)
{
    const std::tuple<at::Tensor, at::Tensor>& second_arg = (h0 == nullptr || c0 == nullptr) ? std::make_tuple(at::Tensor(), at::Tensor()) : std::make_tuple(*h0, *c0);

    Tensor output = nullptr;
    *h_n = nullptr;
    *c_n = nullptr;
    CATCH(
        auto result = (*module)->as<torch::nn::LSTM>()->forward(*input1, second_arg);
        output = new torch::Tensor(std::get<0>(result));
        *h_n = new torch::Tensor(std::get<0>(std::get<1>(result)));
        *c_n = new torch::Tensor(std::get<1>(std::get<1>(result)));
    );
    return output;
}

PackedSequence THSNN_LSTM_forward_with_packed_input(const NNModule module, const PackedSequence input1, const Tensor h0, const Tensor c0, Tensor* h_n, Tensor* c_n)
{
    const std::tuple<at::Tensor, at::Tensor>& second_arg = (h0 == nullptr || c0 == nullptr) ? std::make_tuple(at::Tensor(), at::Tensor()) : std::make_tuple(*h0, *c0);

    PackedSequence output = nullptr;
    *h_n = nullptr;
    *c_n = nullptr;
    CATCH(
        auto result = (*module)->as<torch::nn::LSTM>()->forward_with_packed_input(*input1, second_arg);
        output = new torch::nn::utils::rnn::PackedSequence(std::get<0>(result));
        *h_n = new torch::Tensor(std::get<0>(std::get<1>(result)));
        *c_n = new torch::Tensor(std::get<1>(std::get<1>(result)));
    );
    return output;
}

void THSNN_LSTM_flatten_parameters(const NNModule module)
{
    CATCH(
        (*module)->as<torch::nn::LSTM>()->flatten_parameters();
    );
}

NNModule THSNN_RNNCell_ctor(const int64_t input_size, const int64_t hidden_size, const int64_t nonlinearity, const bool bias, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::RNNCellOptions(input_size, hidden_size)
            .bias(bias);

        ApplyRnnActivation(opts, nonlinearity);

        res = create_module<torch::nn::RNNCellImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_RNNCell_forward(const NNModule module, const Tensor input1, const Tensor h0)
{
    CATCH_TENSOR((*module)->as<torch::nn::RNNCell>()->forward(*input1, (h0 ? *h0 : at::Tensor())));
}

Tensor THSNN_RNNCell_bias_ih(const NNModule module)
{
    return get_bias_ih<torch::nn::RNNCell>(module);
}

void THSNN_RNNCell_set_bias_ih(const NNModule module, const Tensor bias)
{
    set_bias_ih<torch::nn::RNNCell>(module, bias);
}

Tensor THSNN_RNNCell_weight_ih(const NNModule module)
{
    return get_weight_ih<torch::nn::RNNCell>(module);
}

void THSNN_RNNCell_set_weight_ih(const NNModule module, const Tensor weight)
{
    set_weight_ih<torch::nn::RNNCell>(module, weight);
}

Tensor THSNN_RNNCell_bias_hh(const NNModule module)
{
    return get_bias_hh<torch::nn::RNNCell>(module);
}

void THSNN_RNNCell_set_bias_hh(const NNModule module, const Tensor bias)
{
    set_bias_hh<torch::nn::RNNCell>(module, bias);
}

Tensor THSNN_RNNCell_weight_hh(const NNModule module)
{
    return get_weight_hh<torch::nn::RNNCell>(module);
}

void THSNN_RNNCell_set_weight_hh(const NNModule module, const Tensor weight)
{
    set_weight_hh<torch::nn::RNNCell>(module, weight);
}

NNModule THSNN_GRUCell_ctor(const int64_t input_size, const int64_t hidden_size, const bool bias, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::GRUCellOptions(input_size, hidden_size)
        .bias(bias);

        res = create_module<torch::nn::GRUCellImpl>(opts, outAsAnyModule);
    );
}

Tensor  THSNN_GRUCell_forward(const NNModule module, const Tensor input1, const Tensor h0)
{
    CATCH_TENSOR((*module)->as<torch::nn::GRUCell>()->forward(*input1, (h0 ? *h0 : at::Tensor())));
}

Tensor THSNN_GRUCell_bias_ih(const NNModule module)
{
    return get_bias_ih<torch::nn::GRUCell>(module);
}

void THSNN_GRUCell_set_bias_ih(const NNModule module, const Tensor bias)
{
    set_bias_ih<torch::nn::GRUCell>(module, bias);
}

Tensor THSNN_GRUCell_weight_ih(const NNModule module)
{
    return get_weight_ih<torch::nn::GRUCell>(module);
}

void THSNN_GRUCell_set_weight_ih(const NNModule module, const Tensor weight)
{
    set_weight_ih<torch::nn::GRUCell>(module, weight);
}

Tensor THSNN_GRUCell_bias_hh(const NNModule module)
{
    return get_bias_hh<torch::nn::GRUCell>(module);
}

void THSNN_GRUCell_set_bias_hh(const NNModule module, const Tensor bias)
{
    set_bias_hh<torch::nn::GRUCell>(module, bias);
}

Tensor THSNN_GRUCell_weight_hh(const NNModule module)
{
    return get_weight_hh<torch::nn::GRUCell>(module);
}

void THSNN_GRUCell_set_weight_hh(const NNModule module, const Tensor weight)
{
    set_weight_hh<torch::nn::GRUCell>(module, weight);
}

NNModule THSNN_LSTMCell_ctor(const int64_t input_size, const int64_t hidden_size, const bool bias, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::LSTMCellOptions(input_size, hidden_size)
        .bias(bias);

        res = create_module<torch::nn::LSTMCellImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_LSTMCell_forward(const NNModule module, const Tensor input1, const Tensor h0, const Tensor c0, Tensor* c_n)
{
    const std::tuple<at::Tensor, at::Tensor>& second_arg = (h0 == nullptr || c0 == nullptr) ? std::make_tuple(at::Tensor(), at::Tensor()) : std::make_tuple(*h0, *c0);

    Tensor output;
    CATCH(
        auto result = (*module)->as<torch::nn::LSTMCell>()->forward(*input1, second_arg);
        output = new torch::Tensor(std::get<0>(result));
        *c_n = new torch::Tensor(std::get<1>(result));
    );

    return output;
}

Tensor THSNN_LSTMCell_bias_ih(const NNModule module)
{
    return get_bias_ih<torch::nn::LSTMCell>(module);
}

void THSNN_LSTMCell_set_bias_ih(const NNModule module, const Tensor bias)
{
    set_bias_ih<torch::nn::LSTMCell>(module, bias);
}

Tensor THSNN_LSTMCell_weight_ih(const NNModule module)
{
    return get_weight_ih<torch::nn::LSTMCell>(module);
}

void THSNN_LSTMCell_set_weight_ih(const NNModule module, const Tensor weight)
{
    set_weight_ih<torch::nn::LSTMCell>(module, weight);
}

Tensor THSNN_LSTMCell_bias_hh(const NNModule module)
{
    return get_bias_hh<torch::nn::LSTMCell>(module);
}

void THSNN_LSTMCell_set_bias_hh(const NNModule module, const Tensor bias)
{
    set_bias_hh<torch::nn::LSTMCell>(module, bias);
}

Tensor THSNN_LSTMCell_weight_hh(const NNModule module)
{
    return get_weight_hh<torch::nn::LSTMCell>(module);
}

void THSNN_LSTMCell_set_weight_hh(const NNModule module, const Tensor weight)
{
    set_weight_hh<torch::nn::LSTMCell>(module, weight);
}


NNModule THSNN_Sequential_ctor( /* NNAnyModule *submodules, const int length */ )
{
    //std::vector<torch::nn::NamedAnyModule> modules;
    //for (int i = 0; i < length; i++)
    //{
    //	modules.push_back(*(*submodules[i])->as<torch::nn::NamedAnyModule>());
    //}

    auto mod = std::make_shared<torch::nn::SequentialImpl>( /* std::begin(modules), std::end(modules) */ );
    return new std::shared_ptr<torch::nn::Module>(mod);
}

void THSNN_Sequential_push_back(const NNModule module, const char *name, const NNAnyModule submodule)
{
    CATCH (
        (*module)->as<torch::nn::Sequential>()->push_back(name, *(*submodule));
    )
}

Tensor THSNN_Sequential_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Sequential>()->forward(*tensor));
}

Tensor THSNN_one_hot(const Tensor self, const int64_t num_classes)
{
    CATCH_RETURN_Tensor(
        res = ResultTensor(torch::nn::functional::one_hot(*self, num_classes));
    )
}

Tensor THSNN_PackedSequence_data(PackedSequence sequence)
{
    CATCH_TENSOR(sequence->data());
}

Tensor THSNN_PackedSequence_batch_sizes(PackedSequence sequence)
{
    CATCH_TENSOR(sequence->batch_sizes());
}

Tensor THSNN_PackedSequence_sorted_indices(PackedSequence sequence)
{
    CATCH_TENSOR(sequence->sorted_indices());
}

Tensor THSNN_PackedSequence_unsorted_indices(PackedSequence sequence)
{
    CATCH_TENSOR(sequence->unsorted_indices());
}

void THSNN_PackedSequence_dispose(PackedSequence sequence)
{
    delete sequence;
}

PackedSequence THSNN_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first, bool enforce_sorted)
{
    CATCH_RETURN(
        torch::nn::utils::rnn::PackedSequence*,
        nullptr,
        new torch::nn::utils::rnn::PackedSequence(
            torch::nn::utils::rnn::pack_padded_sequence(
                *input, *lengths, batch_first, enforce_sorted)));
}

void THSNN_pad_packed_sequence(PackedSequence sequence, bool batch_first, double padding_value, int64_t total_length, Tensor* res1_, Tensor* res2_)
{
    Tensor& res1 = *res1_;
    Tensor& res2 = *res2_;
    CATCH_TENSORS_2(
        torch::nn::utils::rnn::pad_packed_sequence(
            *sequence, batch_first, padding_value,
            total_length == -1 ? torch::nullopt : c10::optional<int64_t>(total_length)));
}

Tensor THSNN_pad_sequence(const Tensor* sequences, const int sequences_len, bool batch_first, double padding_value)
{
    CATCH_TENSOR(
        torch::nn::utils::rnn::pad_sequence(
            toTensors<at::Tensor>((torch::Tensor**)sequences, sequences_len),
            batch_first, padding_value));
}

PackedSequence THSNN_pack_sequence(const Tensor* sequences, int sequences_len, bool enforce_sorted)
{
    CATCH_RETURN(
        torch::nn::utils::rnn::PackedSequence*,
        nullptr,
        new torch::nn::utils::rnn::PackedSequence(
            torch::nn::utils::rnn::pack_sequence(
                toTensors<at::Tensor>((torch::Tensor**)sequences, sequences_len),
                enforce_sorted)));
}

Tensor THSNN_fold(const Tensor input, const int64_t out1, const int64_t out2, const int64_t kernel1, const int64_t kernel2, const int64_t stride1, const int64_t stride2, const int64_t pad1, const int64_t pad2, const int64_t dil1, const int64_t dil2)
{
    auto opts =
        torch::nn::functional::FoldFuncOptions({ out1, out2 }, {kernel1, kernel2})
        .dilation({dil1, dil2})
        .padding({pad1, pad2})
        .stride({ stride1, stride2 });

    CATCH_TENSOR(torch::nn::functional::fold(*input, opts));
}



Tensor THSNN_unfold(const Tensor input, const int64_t kernel1, const int64_t kernel2, const int64_t stride1, const int64_t stride2, const int64_t pad1, const int64_t pad2, const int64_t dil1, const int64_t dil2)
{
    auto opts =
        torch::nn::functional::UnfoldFuncOptions({ kernel1, kernel2 })
        .dilation({ dil1, dil2 })
        .padding({ pad1, pad2 })
        .stride({ stride1, stride2 });

    CATCH_TENSOR(torch::nn::functional::unfold(*input, opts));
}