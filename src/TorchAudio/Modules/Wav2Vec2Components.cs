// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchaudio,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/audio/blob/76fca37ac8941b72a509a6e58d623632efe04543/torchaudio/models/wav2vec2/components.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/audio/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Linq;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torchaudio.models;

#nullable enable
namespace TorchSharp.Modules
{
    public partial class Wav2Vec2Model : nn.Module<Tensor, Tensor?, (Tensor, Tensor?)>
    {
        /// <summary>
        /// Layer norm with transpose
        /// </summary>
        private class LayerNorm : Module<Tensor, Tensor>
        {
            public readonly long[] normalized_shape;
            public readonly Parameter weight;
            public readonly Parameter bias;
            public readonly double eps;

            public LayerNorm(
                string name,
                long[] normalized_shape,
                double eps = 1e-05,
                bool elementwise_affine = true) : base(name)
            {
                this.normalized_shape = normalized_shape;
                this.weight = torch.nn.Parameter(torch.ones(normalized_shape));
                this.bias = torch.nn.Parameter(torch.zeros(normalized_shape));
                this.eps = eps;
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                var x = input.transpose(-2, -1);
                x = nn.functional.layer_norm(x, this.normalized_shape, this.weight, this.bias, this.eps);
                x = x.transpose(-2, -1);
                return x;
            }
        }

        /// <summary>
        /// Convolution unit of FeatureExtractor
        /// </summary>
        private class ConvLayerBlock : Module<Tensor, Tensor?, (Tensor, Tensor?)>
        {
            public readonly Module<Tensor, Tensor> conv;
            public readonly long kernel_size;
            public readonly Module<Tensor, Tensor>? layer_norm;
            public readonly long stride;

            public ConvLayerBlock(
                string name,
                long in_channels,
                long out_channels,
                long kernel_size,
                long stride,
                bool bias,
                Module<Tensor, Tensor>? layer_norm) : base(name)
            {
                this.kernel_size = kernel_size;
                this.stride = stride;
                this.layer_norm = layer_norm;
                this.conv = nn.Conv1d(
                    inputChannel: in_channels,
                    outputChannel: out_channels,
                    kernelSize: kernel_size,
                    stride: stride,
                    bias: bias);
                RegisterComponents();
            }

            /// <param name="x">Shape: ``[batch, in_channels, in_frame]``</param>
            /// <param name="length">Shape ``[batch, ]``</param>
            /// <returns>
            /// Shape ``[batch, out_channels, out_frames]``.
            /// Shape ``[batch, ]``.
            /// </returns>
            public override (Tensor, Tensor?) forward(
                Tensor x,
                Tensor? length)
            {
                x = this.conv.call(x);
                if (this.layer_norm != null) {
                    x = this.layer_norm.call(x);
                }
                x = nn.functional.gelu(x);

                if (length is not null) {
                    length = torch.div(length - this.kernel_size, this.stride, rounding_mode: RoundingMode.floor) + 1;
                    // When input length is 0, the resulting length can be negative. So fix it here.
                    length = torch.maximum(torch.zeros_like(length), length);
                }
                return (x, length);
            }
        }

        /// <summary>
        /// Extract features from audio
        /// </summary>
        internal class FeatureExtractor : Module<Tensor, Tensor?, (Tensor, Tensor?)>
        {
            public readonly ModuleList<Module<Tensor, Tensor?, (Tensor, Tensor?)>> conv_layers;

            /// <param name="name"></param>
            /// <param name="conv_layers">convolution layers</param>
            public FeatureExtractor(
                string name,
                ModuleList<Module<Tensor, Tensor?, (Tensor, Tensor?)>> conv_layers) : base(name)
            {
                this.conv_layers = conv_layers;
                RegisterComponents();
            }

            /// <param name="x">Input Tensor representing a batch of audio, shape: ``[batch, time]``.</param>
            /// <param name="length">Valid length of each input sample. shape: ``[batch, ]``.</param>
            /// <returns>
            /// The resulting feature, shape: ``[batch, frame, feature]``
            /// Valid length of each output sample. shape: ``[batch, ]``.
            /// </returns>
            /// <exception cref="ArgumentException"></exception>
            public override (Tensor, Tensor?) forward(Tensor x, Tensor? length)
            {
                if (x.ndim != 2) {
                    throw new ArgumentException("Expected the input Tensor to be 2D (batch, time), but received {list(x.shape)}");
                }

                x = x.unsqueeze(1);  // (batch, channel==1, frame)
                foreach (var layer in this.conv_layers) {
                    var conv_layer = (ConvLayerBlock)layer;
                    (x, length) = conv_layer.call(x, length);  // (batch, feature, frame)
                }
                x = x.transpose(1, 2);  // (batch, frame, feature)
                return (x, length);
            }
        }

        /// <summary>
        /// Layer that connects FeatureExtractor and Encoder
        /// </summary>
        private class FeatureProjection : Module<Tensor, Tensor>
        {
            public readonly Module<Tensor, Tensor> dropout;
            public readonly Module<Tensor, Tensor> layer_norm;
            public readonly Module<Tensor, Tensor> projection;

            /// <summary>
            /// Projects features to encoder dimension.
            /// </summary>
            /// <param name="name"></param>
            /// <param name="in_features">Input feature dim.</param>
            /// <param name="out_features">Output feature dim.</param>
            /// <param name="dropout">Dropout probability.</param>
            public FeatureProjection(
                string name,
                long in_features,
                long out_features,
                double dropout) : base(name)
            {
                this.layer_norm = nn.LayerNorm(new long[] { in_features });
                this.projection = nn.Linear(
                    in_features,
                    out_features);
                this.dropout = nn.Dropout(dropout);
                RegisterComponents();
            }

            /// <param name="x">Feature Tensor. shape: ``[batch, frame, in_feature]``</param>
            /// <returns>Projected features. ``[batch, frame, out_feature]``.</returns>
            public override Tensor forward(Tensor x)
            {
                x = this.layer_norm.call(x);
                x = this.projection.call(x);
                x = this.dropout.call(x);
                return x;
            }
        }

        /// <summary>
        /// Positional embedding which is placed at the beginning of Transformer.
        /// </summary>
        internal class ConvolutionalPositionalEmbedding : Module<Tensor, Tensor>
        {
            public readonly Module<Tensor, Tensor> conv;
            public readonly long embed_dim;
            public readonly long num_remove;

            /// <param name="name"></param>
            /// <param name="embed_dim">Feature dimension of the input Tensor.</param>
            /// <param name="kernel_size">The number of frames to be use.</param>
            /// <param name="groups">The number of groups in feature dimensions.</param>
            public ConvolutionalPositionalEmbedding(
                string name,
                long embed_dim,
                long kernel_size,
                long groups) : base(name)
            {
                this.embed_dim = embed_dim;
                // TODO: Replace when nn.utils.weight_norm() is supported.
                // https://github.com/dotnet/TorchSharp/issues/357
                // this.conv = nn.Conv1d(inputChannel: embed_dim, outputChannel: embed_dim, kernelSize: kernel_size, padding: kernel_size / 2, groups: groups);
                // this.conv = nn.utils.weight_norm(this.conv, name: "weight", dim: 2);
                this.conv = new WeightNormConv1d(
                    "WeightNormConv1d",
                    in_channels: embed_dim,
                    out_channels: embed_dim,
                    kernel_size: kernel_size,
                    padding: kernel_size / 2,
                    groups: groups);
                this.num_remove = kernel_size % 2 == 0 ? 1 : 0;
                RegisterComponents();
            }

            /// <param name="x">shape ``[batch, frame, feature]``.</param>
            /// <returns>The resulting feature. Shape ``[batch, frame, feature]``.</returns>
            public override Tensor forward(Tensor x)
            {
                x = x.transpose(-2, -1);
                x = this.conv.call(x);
                if (this.num_remove > 0) {
                    x = x[TensorIndex.Ellipsis, TensorIndex.Slice(null, -this.num_remove)];
                }
                x = torch.nn.functional.gelu(x);
                x = x.transpose(-2, -1);
                return x;
            }

            private class WeightNormConv1d : Module<Tensor, Tensor>
            {
                private readonly Parameter weight_g;
                private readonly Parameter weight_v;
                private readonly Parameter bias;
                private readonly long padding;
                private readonly long groups;

                public WeightNormConv1d(string name, long in_channels, long out_channels, long kernel_size, long padding, long groups) : base(name)
                {
                    this.weight_g = Parameter(
                        torch.empty(
                            new long[] { 1, in_channels / groups, kernel_size },
                            dtype: torch.float32));
                    this.weight_v = Parameter(
                        torch.empty(
                            new long[] { out_channels, in_channels / groups, kernel_size },
                            dtype: torch.float32));
                    this.bias = Parameter(torch.empty(out_channels, dtype: torch.float32));
                    this.padding = padding;
                    this.groups = groups;
                    this.RegisterComponents();
                    this.reset_parameters();
                }

                public void reset_parameters()
                {
                    var weight = torch.empty_like(this.weight_v);
                    init.kaiming_uniform_(weight, Math.Sqrt(5));
                    using (torch.no_grad()) {
                        this.weight_g.set_(torch.sqrt(torch.square(weight).sum(dim: new long[] { 0, 1 }, keepdim: true)));
                        this.weight_v.set_(weight / this.weight_g);
                    }
                    var fan_in = this.weight_v.size(1);
                    var bound = 1.0 / Math.Sqrt(fan_in);
                    init.uniform_(this.bias, -bound, bound);
                }

                public override Tensor forward(Tensor input)
                {
                    var weight_v_norm = torch.linalg.norm(weight_v, dims: new long[] { 0, 1 }, keepdim: true);
                    var weight = torch.mul(weight_v / weight_v_norm, weight_g);
                    return nn.functional.conv1d(input, weight, bias, padding: this.padding, groups: this.groups);
                }
            }
        }

        /// <summary>
        /// Multihead Self Attention module
        /// </summary>
        private class SelfAttention : Module<Tensor, Tensor?, Tensor>
        {
            public readonly Module<Tensor, Tensor> dropout;
            public readonly long embed_dim;
            public readonly long head_dim;
            public readonly Module<Tensor, Tensor> k_proj;
            public readonly long num_heads;
            public readonly Module<Tensor, Tensor> out_proj;
            public readonly Module<Tensor, Tensor> q_proj;
            public readonly double scaling;
            public readonly Module<Tensor, Tensor> v_proj;

            /// <param name="name"></param>
            /// <param name="embed_dim">Total dimension of the model.</param>
            /// <param name="num_heads">The number of heads.</param>
            /// <param name="dropout">Dropout probabiliry on attn_output_weights. Default: ``0.0``</param>
            /// <exception cref="ArgumentException"></exception>
            public SelfAttention(
                string name,
                long embed_dim,
                long num_heads,
                double dropout = 0.0) : base(name)
            {
                var head_dim = embed_dim / num_heads;
                if (head_dim * num_heads != embed_dim) {
                    throw new ArgumentException($"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`");
                }
                this.embed_dim = embed_dim;
                this.num_heads = num_heads;
                this.dropout = torch.nn.Dropout(dropout);
                this.head_dim = head_dim;

                this.scaling = Math.Pow(this.head_dim, -0.5);

                this.k_proj = nn.Linear(embed_dim, embed_dim, hasBias: true);
                this.v_proj = nn.Linear(embed_dim, embed_dim, hasBias: true);
                this.q_proj = nn.Linear(embed_dim, embed_dim, hasBias: true);
                this.out_proj = nn.Linear(embed_dim, embed_dim, hasBias: true);
                RegisterComponents();
            }

            /// <param name="x">shape: ``[batch_size, sequence_length, embed_dim]``.</param>
            /// <param name="attention_mask">shape: ``[batch_size, 1, sequence_length, sequence_length]``</param>
            /// <returns>The resulting tensor. shape: ``[batch, sequence_length, embed_dim]``</returns>
            /// <exception cref="ArgumentException"></exception>
            public override Tensor forward(Tensor x, Tensor? attention_mask)
            {
                if (x.ndim != 3 || x.shape[2] != this.embed_dim) {
                    throw new ArgumentException("The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). Found {x.shape}.");
                }
                var batch_size = x.size(0);
                var length = x.size(1);
                var embed_dim = x.size(2);
                if (attention_mask is not null) {
                    var shape_ = new long[] { batch_size, 1, length, length };
                    if (attention_mask.dim() != shape_.Length ||
                        attention_mask.size(0) != shape_[0] ||
                        attention_mask.size(1) != shape_[1] ||
                        attention_mask.size(2) != shape_[2] ||
                        attention_mask.size(3) != shape_[3]) {
                        throw new ArgumentException($"The expected attention mask shape is {shape_}. Found {attention_mask.size()}.");
                    }
                }

                var shape = new long[] { batch_size, length, this.num_heads, this.head_dim };
                var q = this.q_proj.call(x).view(shape).transpose(2, 1);  // B, nH, L, Hd
                var k = this.k_proj.call(x).view(shape).permute(0, 2, 3, 1);  // B, nH, Hd, L
                var v = this.v_proj.call(x).view(shape).transpose(2, 1);  // B, nH, L, Hd

                var weights = this.scaling * torch.matmul(q, k);  // B, nH, L, L
                if (attention_mask is not null) {
                    weights += attention_mask;
                }

                weights = torch.nn.functional.softmax(weights, dim: -1);
                weights = this.dropout.call(weights);

                var output = torch.matmul(weights, v);  // B, nH, L, Hd
                output = output.transpose(2, 1).reshape(batch_size, length, embed_dim);

                output = this.out_proj.call(output);
                return output;
            }

            public new Tensor call(Tensor x, Tensor? attention_mask = null)
            {
                return base.call(x, attention_mask);
            }
        }

        /// <summary>
        /// Layer that follows attention layer in encoder layer.
        /// </summary>
        private class FeedForward : Module<Tensor, Tensor>
        {
            public readonly Module<Tensor, Tensor> intermediate_dense;
            public readonly Module<Tensor, Tensor> intermediate_dropout;
            public readonly Module<Tensor, Tensor> output_dense;
            public readonly Module<Tensor, Tensor> output_dropout;

            public FeedForward(
                string name,
                long io_features,
                long intermediate_features,
                double intermediate_dropout,
                double output_dropout) : base(name)
            {
                this.intermediate_dense = nn.Linear(io_features, intermediate_features);
                this.intermediate_dropout = nn.Dropout(intermediate_dropout);
                this.output_dense = nn.Linear(intermediate_features, io_features);
                this.output_dropout = nn.Dropout(output_dropout);
                RegisterComponents();
            }

            /// <param name="x">shape: `(batch, sequence_length, io_features)`</param>
            /// <returns>shape: `(batch, sequence_length, io_features)`</returns>
            public override Tensor forward(Tensor x)
            {
                x = this.intermediate_dense.call(x);
                x = torch.nn.functional.gelu(x);
                x = this.intermediate_dropout.call(x);

                x = this.output_dense.call(x);
                x = this.output_dropout.call(x);
                return x;
            }
        }

        /// <summary>
        /// A layer unit in encoder. Combines multihead self attention and feed forward.
        /// </summary>
        private class EncoderLayer : Module<Tensor, Tensor?, Tensor>
        {
            public readonly SelfAttention attention;
            public readonly Module<Tensor, Tensor> dropout;
            public readonly Module<Tensor, Tensor> feed_forward;
            public readonly Module<Tensor, Tensor> final_layer_norm;
            public readonly Module<Tensor, Tensor> layer_norm;
            public bool layer_norm_first;

            public EncoderLayer(
                string name,
                SelfAttention attention,
                double dropout,
                bool layer_norm_first,
                Module<Tensor, Tensor> feed_forward) : base(name)
            {
                this.attention = attention;
                this.dropout = nn.Dropout(dropout);
                this.layer_norm = nn.LayerNorm(new long[] { attention.embed_dim });
                this.layer_norm_first = layer_norm_first;
                this.feed_forward = feed_forward;
                this.final_layer_norm = nn.LayerNorm(new long[] { attention.embed_dim });
                RegisterComponents();
            }

            /// <param name="x">shape: `(batch, sequence_length, embed_dim)`</param>
            /// <param name="attention_mask">shape: `(batch, 1, sequence_length, sequence_length)`</param>
            /// <returns></returns>
            public override Tensor forward(
                Tensor x,
                Tensor? attention_mask = null)
            {
                var residual = x;

                if (this.layer_norm_first) {
                    x = this.layer_norm.call(x);
                }

                x = this.attention.call(x, attention_mask);
                x = this.dropout.call(x);
                x = residual + x;

                if (this.layer_norm_first) {
                    x = x + this.feed_forward.call(this.final_layer_norm.call(x));
                } else {
                    x = this.layer_norm.call(x);
                    x = this.final_layer_norm.call(x + this.feed_forward.call(x));
                }
                return x;
            }
        }

        internal class Transformer : Module<Tensor, Tensor?, Tensor>
        {
            public readonly Module<Tensor, Tensor> dropout;
            public readonly double layer_drop;
            public readonly Module<Tensor, Tensor> layer_norm;
            public readonly bool layer_norm_first;
            public readonly ModuleList<Module<Tensor, Tensor?, Tensor>> layers;

            public ConvolutionalPositionalEmbedding pos_conv_embed;

            public Transformer(
                string name,
                ConvolutionalPositionalEmbedding pos_conv_embed,
                double dropout,
                ModuleList<Module<Tensor, Tensor?, Tensor>> layers,
                bool layer_norm_first,
                double layer_drop) : base(name)
            {
                this.pos_conv_embed = pos_conv_embed;
                this.layer_norm = nn.LayerNorm(new long[] { pos_conv_embed.embed_dim });
                this.layer_norm_first = layer_norm_first;
                this.layer_drop = layer_drop;
                this.dropout = nn.Dropout(dropout);
                this.layers = layers;
                RegisterComponents();
            }

            public Tensor _preprocess(Tensor x)
            {
                x = x + this.pos_conv_embed.call(x);

                if (this.layer_norm_first) {
                    x = this.layer_norm.call(x);
                }

                x = this.dropout.call(x);
                return x;
            }

            public override Tensor forward(
                Tensor x,
                Tensor? attention_mask = null)
            {
                x = this._preprocess(x);
                foreach (var layer in this.layers) {
                    if (!(this.training && torch.rand(1).item<float>() <= this.layer_drop)) {
                        x = ((nn.Module<Tensor, Tensor?, Tensor>)layer).call(x, attention_mask);
                    }
                }

                if (!this.layer_norm_first) {
                    x = this.layer_norm.call(x);
                }

                return x;
            }

            public new Tensor call(Tensor x, Tensor? attention_mask = null) => base.call(x, attention_mask);

            public Tensor[] get_intermediate_outputs(
                Tensor x,
                Tensor? attention_mask = null,
                int? num_layers = null)
            {
                if (num_layers != null) {
                    if (!(0 < num_layers && num_layers <= this.layers.Count)) {
                        throw new ArgumentException($"`num_layers` must be between [1, {this.layers.Count}]");
                    }
                }
                var ret = new List<Tensor>();
                x = this._preprocess(x);
                foreach (var layer in this.layers) {
                    x = ((nn.Module<Tensor, Tensor?, Tensor>)layer).call(x, attention_mask);
                    ret.Add(x);
                    if (num_layers != null && ret.Count >= num_layers) {
                        return ret.ToArray();
                    }
                }
                return ret.ToArray();
            }
        }

        internal class Encoder : Module<Tensor, Tensor?, Tensor>
        {
            public readonly Module<Tensor, Tensor> feature_projection;
            public readonly Transformer transformer;

            public Encoder(
                string name,
                Module<Tensor, Tensor> feature_projection,
                Transformer transformer) : base(name)
            {
                this.feature_projection = feature_projection;
                this.transformer = transformer;
                RegisterComponents();
            }

            public (Tensor, Tensor?) _preprocess(
                Tensor features,
                Tensor? lengths = null)
            {
                var x = this.feature_projection.call(features);

                Tensor? mask = null;
                if (lengths is not null) {
                    var batch_size = x.size(0);
                    var max_len = x.size(1);
                    // create mask for padded elements and zero-out them
                    mask = torch.arange(max_len, device: lengths.device).expand(batch_size, max_len) >= lengths[TensorIndex.Colon, TensorIndex.None];
                    x[mask] = 0.0;
                    // extend the mask to attention shape and set weight
                    mask = -10000.0 * mask[TensorIndex.Colon, TensorIndex.None, TensorIndex.None, TensorIndex.Colon].to(type: features.dtype);
                    mask = mask.expand(batch_size, 1, max_len, max_len);
                }
                return (x, mask);
            }

            public override Tensor forward(
                Tensor features,
                Tensor? lengths = null)
            {
                var (x, mask) = this._preprocess(features, lengths);
                x = this.transformer.call(x, attention_mask: mask);
                return x;
            }

            public Tensor[] extract_features(
                Tensor features,
                Tensor? lengths = null,
                int? num_layers = null)
            {
                var (x, masks) = this._preprocess(features, lengths);
                return this.transformer.get_intermediate_outputs(x, attention_mask: masks, num_layers: num_layers);
            }
        }

        /// <summary>
        /// See Also:
        /// * Original implementation
        /// https:///github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L666-L733
        /// * "extractor_mode"
        /// - Def and base:
        /// https:///github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L38-L45
        /// - Large:
        /// https:///github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L52
        /// * "conv_feature_layers"
        /// - Def, base and large:
        /// https:///github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L94-L100
        /// * "conv_bias"
        /// - Def and base:
        /// https:///github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L101-L103
        /// - Large:
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L61
        /// If "group_norm", then a single normalization is applied
        /// </summary>
        /// <param name="norm_mode">Either "group_norm" or "layer_norm".
        /// If "group_norm", then a single normalization is applied
        /// in the first convolution block. Otherwise, all the convolution
        /// blocks will have layer normalization.
        /// This option corresponds to "extractor_mode" from fairseq.
        /// Expected values are "group_norm" for Base arch, and
        /// "layer_norm" for Large arch.</param>
        /// <param name="shapes">Configuration of convolution layers. List of convolution configuration,
        /// i.e. ``[(output_channel, kernel_size, stride), ...]``
        /// This option corresponds to "conv_feature_layers" from fairseq.
        /// Expected values are
        /// ``[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2``
        /// for all the architectures.</param>
        /// <param name="bias">Whether to include bias term to each convolution operation.
        /// This option corresponds to "conv_bias" from fairseq.
        /// Expected values are False for Base arch, and True for Large arch.</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        internal static FeatureExtractor _get_feature_extractor(FeatureExtractorNormMode norm_mode, long[][] shapes, bool bias)
        {
            var blocks = ModuleList<Module<Tensor, Tensor?, (Tensor, Tensor?)>> ();
            long in_channels = 1;
            for (int i = 0; i < shapes.Length; i++) {
                var shape = shapes[i];
                var out_channels = shape[0];
                var kernel_size = shape[1];
                var stride = shape[2];
                Module<Tensor, Tensor>? normalization = null;
                if (norm_mode == FeatureExtractorNormMode.group_norm && i == 0) {
                    normalization = nn.GroupNorm(
                        num_groups: out_channels,
                        num_channels: out_channels,
                        affine: true);
                } else if (norm_mode == FeatureExtractorNormMode.layer_norm) {
                    normalization = new Wav2Vec2Model.LayerNorm(
                        "LayerNorm",
                        normalized_shape: new long[] { out_channels },
                        elementwise_affine: true);
                }
                blocks.Add(
                    new ConvLayerBlock(
                        "ConvlayerBlock",
                        in_channels: in_channels,
                        out_channels: out_channels,
                        kernel_size: kernel_size,
                        stride: stride,
                        bias: bias,
                        layer_norm: normalization));
                in_channels = out_channels;
            }
            return new FeatureExtractor("FeatureExtractor", blocks);
        }

        /// <summary>
        /// See Also:
        /// * "encoder_embed_dim"
        /// - Def and base
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L49-L51
        /// - Large
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L64
        /// * "dropout_input"
        /// - Def, base and large
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L75-L78
        /// * "conv_pos"
        /// - Def, base and large
        /// NOTE: The description is wrong.
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L204-L207
        /// - Usage
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L756
        /// * "conv_pos_groups"
        /// - Def, base and large
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L208-L211
        /// * "encoder_layers"
        /// - Def and base
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L46-L48
        /// - Large
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L63
        /// * "encoder_attention_heads"
        /// - Def and base
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L55-L57
        /// - Large
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L66
        /// * "attention_dropout"
        /// - Def and base
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L66-L68
        /// - Large
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L60
        /// * "encoder_ffn_embed_dim"
        /// - Def and base
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L52-L54
        /// - Large
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L65
        /// * "activation_dropout"
        /// - Def
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L69-L71
        /// - Base
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/base_960h.yaml#L55
        /// - Large
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/vox_960h.yaml#L55
        /// * "dropout"
        /// - Def and base
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L63-L65
        /// - Large
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L59
        /// * "layer_norm_first"
        /// - Def and base
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L91-L93
        /// - Large
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L53
        /// * "layerdrop"
        /// - Def
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L72-L74
        /// - Base
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/base_960h.yaml#L54
        /// - Large
        /// https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/vox_960h.yaml#L54
        /// </summary>
        /// <param name="in_features">in_features (int): The number of input features.</param>
        /// <param name="embed_dim">The dimension of embedding.
        /// This option corresponds to "encoder_embed_dim" from fairseq.
        /// Expected values are 768 for Base arch, and 1024 for Large arch.</param>
        /// <param name="dropout_input">The dropout probability applied after the input feature is projected
        /// to ``embed_dim``.
        /// This option corresponds to "dropout_input" from fairseq.
        /// Expected values are 0.1 for both Base and Large arch.</param>
        /// <param name="pos_conv_kernel">The kernel size of convolutional positional embeddings.
        /// This option corresponds to "conv_pos" from fairseq.
        /// Expected values are 128 for both Base and Large arch.</param>
        /// <param name="pos_conv_groups">The number of groups of convolutional positional embeddings.
        /// This option corresponds to "conv_pos_groups" from fairseq.
        /// Expected values are 16 for both Base and Large arch.</param>
        /// <param name="num_layers">The number of self attention layers in transformer block.
        /// This option corresponds to "encoder_layers" from fairseq.
        /// Expected values are 12 for Base and 24 for Large arch.</param>
        /// <param name="num_heads">The number of heads in self attention layers.
        /// This option corresponds to "encoder_attention_heads" from fairseq.
        /// Expected values are 12 for Base and 16 for Large arch.</param>
        /// <param name="attention_dropout">The dropout probability applied after softmax in self-attention layer.
        /// This option corresponds to "attention_dropout" from fairseq.
        /// Expected values are 0.1 for Base and 0.0 for Large arch.</param>
        /// <param name="ff_interm_features">The dimension of hidden features in feed forward layer.
        /// This option corresponds to "encoder_ffn_embed_dim" from fairseq.
        /// Expected values are 3072 for Base and 4096 for Large arch.</param>
        /// <param name="ff_interm_dropout">The dropout probability applied in feedforward layer.
        /// This option correspinds to "activation_dropout" from fairseq.
        /// Expected values are 0.1 for both Base and Large arch.</param>
        /// <param name="dropout">The dropout probability applied at the end of feed forward layer.
        /// This option corresponds to "dropout" from fairseq.
        /// Expected values are 0.1 for Base and 0.0 for Large arch.</param>
        /// <param name="layer_norm_first">Control the order of layer norm in transformer layer and each encoder layer.
        /// If True, in transformer layer, layer norm is applied before features are fed
        /// to encoder layers. In encoder layer, two layer norms are applied before and after
        /// self attention.
        /// If False, in transformer layer, layer norm is applied after features are fed
        /// to encoder layers. In encoder layer, two layer norms are applied after self
        /// attention, before and after feed forward.
        /// This option corresponds to "layer_norm_first" from fairseq.
        /// Expected values are False for Base and True for Large arch.</param>
        /// <param name="layer_drop">Probability to drop each encoder layer during training.
        /// This option corresponds to "layerdrop" from fairseq.
        /// Expected values are 0.1 for both Base and Large arch.</param>
        /// <returns></returns>
        internal static Encoder _get_encoder(
            long in_features,
            long embed_dim,
            double dropout_input,
            long pos_conv_kernel,
            long pos_conv_groups,
            long num_layers,
            long num_heads,
            double attention_dropout,
            long ff_interm_features,
            double ff_interm_dropout,
            double dropout,
            bool layer_norm_first,
            double layer_drop)
        {
            var feature_projection = new FeatureProjection("featureprojection", in_features, embed_dim, dropout_input);
            var pos_conv = new ConvolutionalPositionalEmbedding("convolutionalpositionalembedding", embed_dim, pos_conv_kernel, pos_conv_groups);

            // Original impl
            // https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L768-L782
            var encoder_layers = ModuleList<Module<Tensor, Tensor?, Tensor>>();
            for (long i = 0; i < num_layers; i++) {
                var attention = new SelfAttention(
                    "SelfAttention",
                    embed_dim: embed_dim,
                    num_heads: num_heads,
                    dropout: attention_dropout);
                var feed_forward = new FeedForward(
                    "FeedForward",
                    io_features: embed_dim,
                    intermediate_features: ff_interm_features,
                    intermediate_dropout: ff_interm_dropout,
                    output_dropout: dropout);
                encoder_layers.append(
                    new EncoderLayer(
                        "EncoderLayer",
                        attention: attention,
                        dropout: dropout,
                        layer_norm_first: layer_norm_first,
                        feed_forward: feed_forward));
            }
            var transformer = new Transformer(
                "Transformer",
                pos_conv_embed: pos_conv,
                dropout: dropout,
                layers: encoder_layers,
                layer_norm_first: !layer_norm_first,
                layer_drop: layer_drop);
            return new Encoder("Encoder", feature_projection, transformer);
        }

        /// <summary>
        /// Computes random mask spans for a given shape.
        /// </summary>
        /// <param name="shape">The shape for which to compute masks.
        /// The first element is batch size and second is the number of frames.</param>
        /// <param name="padding_mask">The padding mask of the same dimension as shape,
        /// which will prevent masking padded elements.</param>
        /// <param name="mask_prob">Probability for each token to be chosen as start of the span to be masked.
        /// This will be multiplied by number of timesteps divided by length of mask span to mask
        /// approximately this percentage of all elements. However due to overlaps, the actual number
        /// will be smaller (unless no_overlap is True).</param>
        /// <param name="mask_length"></param>
        /// <param name="mask_type">How to compute mask lengths. Options: [``static``, ``uniform``, ``normal``, ``poisson``].
        /// ``static``: Fixed size
        /// ``uniform``: Sample from uniform distribution [mask_other, mask_length*2]
        /// ``normal``: Sample from normal distribution with mean ``mask_length`` and stdev ``mask_other``.
        /// ``poisson``: Sample from possion distribution with lambda = ``mask_length``.</param>
        /// <param name="mask_other"></param>
        /// <param name="min_masks">Minimum number of masked spans.</param>
        /// <param name="no_overlap">If false, will switch to an alternative recursive algorithm
        /// that prevents spans from overlapping.</param>
        /// <param name="min_space">How many frames to keep unmasked between spans (Only used if no_overlap is true).</param>
        /// <returns>The mask indices of dimension `[batch, frame]`.</returns>
        /// <exception cref="Exception"></exception>
        private static Tensor _compute_mask_indices(
            long[] shape,
            Tensor? padding_mask,
            double mask_prob,
            long mask_length,
            string mask_type = "static",
            double mask_other = 0.0,
            long min_masks = 0,
            bool no_overlap = false,
            long min_space = 0)
        {
            long min_len;
            var batch_size = shape[0];
            var frame = shape[1];
            var mask = torch.full(new long[] { batch_size, frame }, false);
            // add a random number for probabilistic rounding
            var all_num_mask = (long)(mask_prob * frame / mask_length + torch.rand(1));

            all_num_mask = Math.Max(min_masks, all_num_mask);

            var mask_idcs = new List<Tensor>();
            for (long i = 0; i < batch_size; i++) {
                Tensor mask_idc;
                Tensor lengths;
                long num_mask;
                long sz;

                if (padding_mask is not null) {
                    sz = frame - padding_mask[i].@long().sum().item<long>();
                    // add a random number for probabilistic rounding
                    num_mask = (long)((mask_prob * sz / mask_length) + torch.rand(1));
                    num_mask = Math.Max(min_masks, num_mask);
                } else {
                    sz = frame;
                    num_mask = all_num_mask;
                }

                if (mask_type == "static") {
                    lengths = torch.full(new long[] { num_mask }, mask_length);
                } else if (mask_type == "uniform") {
                    lengths = torch.randint((long)mask_other, mask_length * 2 + 1, size: new long[] { num_mask });
                } else if (mask_type == "normal") {
                    lengths = torch.normal(mask_length, mask_other, size: new long[] { num_mask });
                    lengths = torch.maximum(torch.ones(1), torch.round(lengths)).@int();
                } else if (mask_type == "poisson") {
                    // torch.poisson() of PyTorch doesn't accept the argument `size'.
                    throw new Exception($"unsupported mask selection: {mask_type}");
                    // lengths = torch.poisson(mask_length, size: new long[] { num_mask });
                    // lengths = torch.round(lengths).@int();
                } else {
                    throw new Exception($"unknown mask selection: {mask_type}");
                }

                if (lengths.sum().item<int>() == 0) {
                    lengths[0] = Math.Min(mask_length, sz - 1);
                }

                if (no_overlap) {
                    var mask_idc_data = new List<long>();

                    List<(long, long)> arrange(long s, long e, long length, long keep_length)
                    {
                        long span_start = torch.randint(s, e - length, dtype: torch.int64, size: new long[] { 1 }).item<long>();
                        for (long i = 0; i < length; i++) {
                            mask_idc_data.Add(span_start + i);
                        }
                        var new_parts = new List<(long, long)>();
                        if (span_start - s - min_space >= keep_length) {
                            new_parts.Add((s, span_start - min_space + 1));
                        }
                        if (e - span_start - keep_length - min_space > keep_length) {
                            new_parts.Add((span_start + length + min_space, e));
                        }
                        return new_parts;
                    }

                    var parts = new List<(long, long)> { (0, sz) };
                    var min_length = torch.min(lengths).item<long>();
                    foreach (var length in lengths.data<long>().OrderByDescending(x => x)) {
                        var lens = torch.tensor((
                            from p in parts
                            select p.Item2 - p.Item1).ToArray(), dtype: torch.int64);
                        lens[lens < length + min_space] = 0;
                        long l_sum = lens.sum().item<long>();
                        if (l_sum == 0) {
                            break;
                        }
                        var probs = lens / l_sum;
                        var c = (int)torch.distributions.Categorical(probs).sample().item<long>();
                        var (s, e) = parts[c];
                        parts.RemoveAt(c);
                        parts.AddRange(arrange(s, e, length, min_length));
                    }
                    mask_idc = torch.tensor(mask_idc_data.ToArray());
                } else {
                    min_len = torch.min(lengths).item<long>();
                    if (sz - min_len <= num_mask) {
                        min_len = sz - num_mask - 1;
                    }

                    mask_idc = torch.multinomial(torch.ones(new long[] { sz - min_len }), num_samples: num_mask, replacement: false);

                    mask_idc = torch.tensor((
                        from j in Enumerable.Range(0, (int)mask_idc.size(0))
                        from offset in Enumerable.Range(0, (int)lengths[j].item<long>())
                        select (mask_idc[j].item<long>() + offset)).ToArray());
                }

                mask_idcs.Add(mask_idc[mask_idc < sz].unique().Item1);
            }

            min_len = (from m in mask_idcs select m.size(0)).Min();
            for (int i = 0; i < mask_idcs.Count; i++) {
                var mask_idc = mask_idcs[i];
                if (mask_idc.size(0) > min_len) {
                    mask_idc = torch.index_select(
                        mask_idc,
                        0,
                        torch.multinomial(
                            torch.ones(new long[] { mask_idc.shape[0] }),
                            num_samples: min_len,
                            replacement: false));
                }
                mask[i, mask_idc] = true;
            }

            return mask;
        }

        /// <summary>
        /// Generate the padding mask given the padded input and the lengths Tensors.
        /// </summary>
        /// <param name="input">The padded Tensor of dimension `[batch, max_len, frequency]`.</param>
        /// <param name="lengths">The lengths Tensor of dimension `[batch,]`.</param>
        /// <returns>The padding mask.</returns>
        internal static Tensor _get_padding_mask(Tensor input, Tensor lengths)
        {
            var batch_size = input.size(0);
            var max_len = input.size(1);
            var mask = torch.arange(max_len, device: lengths.device).expand(batch_size, max_len) >= lengths[TensorIndex.Colon, TensorIndex.None];
            return mask;
        }

        /// <summary>
        /// Generate the masks for masked prediction.
        /// </summary>
        internal class MaskGenerator : Module<Tensor, Tensor?, (Tensor, Tensor?)>
        {
            public readonly long mask_channel_length;
            public readonly long mask_channel_min_space;
            public readonly double mask_channel_other;
            public readonly double mask_channel_prob;
            public readonly string mask_channel_selection;
            public readonly Parameter mask_embedding;
            public readonly long mask_length;
            public readonly long mask_min_space;
            public readonly double mask_other;
            public readonly double mask_prob;
            public readonly string mask_selection;
            public readonly bool no_mask_channel_overlap;
            public readonly bool no_mask_overlap;

            /// <param name="name"></param>
            /// <param name="encoder_embed_dim">The dimension of the transformer embedding output.</param>
            /// <param name="mask_prob">Probability for each token to be chosen as start of the span to be masked.
            /// This will be multiplied by number of timesteps divided by length of mask span to mask
            /// approximately this percentage of all elements. However due to overlaps, the actual number
            /// will be smaller (unless no_overlap is True).</param>
            /// <param name="mask_selection">How to choose the mask length.
            /// Options: [``static``, ``uniform``, ``normal``, ``poisson``].</param>
            /// <param name="mask_other">Secondary mask argument (used for more complex distributions).</param>
            /// <param name="mask_length">The lengths of the mask.</param>
            /// <param name="no_mask_overlap">Whether to allow masks to overlap.</param>
            /// <param name="mask_min_space">Minimum space between spans (if no overlap is enabled).</param>
            /// <param name="mask_channel_prob">The probability of replacing a feature with 0.</param>
            /// <param name="mask_channel_selection">How to choose the mask length for channel masking.
            /// Options: [``static``, ``uniform``, ``normal``, ``poisson``].</param>
            /// <param name="mask_channel_other">Secondary mask argument for channel masking(used for more complex distributions).</param>
            /// <param name="mask_channel_length">Minimum space between spans (if no overlap is enabled) for channel masking.</param>
            /// <param name="no_mask_channel_overlap">Whether to allow channel masks to overlap.</param>
            /// <param name="mask_channel_min_space">Minimum space between spans for channel masking(if no overlap is enabled).</param>
            public MaskGenerator(
                string name,
                long encoder_embed_dim,
                double mask_prob,
                string mask_selection,
                double mask_other,
                long mask_length,
                bool no_mask_overlap,
                long mask_min_space,
                double mask_channel_prob,
                string mask_channel_selection,
                double mask_channel_other,
                long mask_channel_length,
                bool no_mask_channel_overlap,
                long mask_channel_min_space) : base(name)
            {
                this.mask_prob = mask_prob;
                this.mask_selection = mask_selection;
                this.mask_other = mask_other;
                this.mask_length = mask_length;
                this.no_mask_overlap = no_mask_overlap;
                this.mask_min_space = mask_min_space;
                this.mask_channel_prob = mask_channel_prob;
                this.mask_channel_selection = mask_channel_selection;
                this.mask_channel_other = mask_channel_other;
                this.mask_channel_length = mask_channel_length;
                this.no_mask_channel_overlap = no_mask_channel_overlap;
                this.mask_channel_min_space = mask_channel_min_space;
                this.mask_embedding = Parameter(torch.empty(encoder_embed_dim, dtype: torch.float32));
                torch.nn.init.uniform_(this.mask_embedding);
                RegisterComponents();
            }

            /// <param name="x">The encoded representations after feature extraction module.</param>
            /// <param name="padding_mask">The padding mask of the same dimension as shape,
            /// which will prevent masking padded elements.</param>
            /// <returns>
            /// The feature representations after masking.
            /// The generated mask indices.
            /// </returns>
            public override (Tensor, Tensor?) forward(Tensor x, Tensor? padding_mask)
            {
                Tensor? mask_indices;
                var B = x.size(0);
                var T = x.size(1);
                var C = x.size(2);
                if (this.mask_prob > 0) {
                    mask_indices = _compute_mask_indices(
                        new long[] { B, T },
                        padding_mask,
                        this.mask_prob,
                        this.mask_length,
                        this.mask_selection,
                        this.mask_other,
                        min_masks: 2,
                        no_overlap: this.no_mask_overlap,
                        min_space: this.mask_min_space);
                    mask_indices = mask_indices.to(x.device);
                    x[mask_indices] = this.mask_embedding;
                } else {
                    mask_indices = null;
                }

                if (this.mask_channel_prob > 0) {
                    var mask_channel_indices = _compute_mask_indices(
                        new long[] { B, C },
                        null,
                        this.mask_channel_prob,
                        this.mask_channel_length,
                        this.mask_channel_selection,
                        this.mask_channel_other,
                        no_overlap: this.no_mask_channel_overlap,
                        min_space: this.mask_channel_min_space);
                    mask_channel_indices = mask_channel_indices.to(x.device).unsqueeze(1).expand(-1, T, -1);
                    x[mask_channel_indices] = 0;
                }

                return (x, mask_indices);
            }
        }

        /// <summary>
        /// Compute the logits of the embeddings.
        /// </summary>
        /// <param name="proj_x">The projected masked representations of dimension `[batch, frame, final_dim]`.</param>
        /// <param name="target">The target Tensor of dimension `[batch, frame, final_dim]`.</param>
        /// <param name="label_embeddings">The trainable embeddings of target of dimension `[num_class, final_dim]`.</param>
        /// <returns>The logits of the inputs.</returns>
        private static Tensor _compute_logits(
            Tensor proj_x,
            Tensor target,
            Tensor label_embeddings)
        {
            var logit_temp = 0.1;
            var pos = torch.index_select(label_embeddings, 0, target.@long());
            var negs = label_embeddings.unsqueeze(1).expand(-1, proj_x.size(0), -1);
            var neg_is_pos = (pos == negs).all(-1);
            pos = pos.unsqueeze(0);
            var targets = torch.cat(new Tensor[] { pos, negs }, dim: 0);

            var logits = torch.nn.functional.cosine_similarity(proj_x.@float(), targets.@float(), dim: -1).type_as(proj_x);
            logits /= logit_temp;
            if (neg_is_pos.any().item<bool>()) {
                logits[1][neg_is_pos] = double.NegativeInfinity;
            }
            logits = logits.transpose(0, 1);  // (num_x, num_cls+1)
            return logits;
        }

        /// <summary>
        /// Generate the logits of masked and unmasked inputs.
        /// </summary>
        internal class LogitGenerator : Module<Tensor, Tensor, Tensor, Tensor, (Tensor?, Tensor?)>
        {
            public readonly Module<Tensor, Tensor> final_proj;
            public readonly Tensor label_embeddings;
            public readonly bool skip_masked;
            public readonly bool skip_nomask;

            /// <param name="name"></param>
            /// <param name="encoder_embed_dim">The dimension of the transformer embedding output.</param>
            /// <param name="num_classes">The number of classes in the labels.</param>
            /// <param name="final_dim">Project final representations and targets to `final_dim`.</param>
            /// <param name="skip_masked">If true, skip computing losses over masked frames.</param>
            /// <param name="skip_nomask">If true, skip computing losses over unmasked frames.</param>
            public LogitGenerator(
                string name,
                long encoder_embed_dim,
                long num_classes,
                long final_dim,
                bool skip_masked,
                bool skip_nomask) : base(name)
            {
                this.label_embeddings = Parameter(torch.empty(new long[] { num_classes, final_dim }, dtype: torch.float32));
                torch.nn.init.uniform_(this.label_embeddings);
                this.final_proj = torch.nn.Linear(encoder_embed_dim, final_dim);
                this.skip_masked = skip_masked;
                this.skip_nomask = skip_nomask;
                RegisterComponents();
            }

            /// <param name="x">The feature representation of the last transformer layer.</param>
            /// <param name="label"> The label Tensor of dimension `[batch, frame]`.</param>
            /// <param name="mask_m">The masked indices of dimension `[batch, frame]`.</param>
            /// <param name="mask_u">The unmasked indices of dimension `[batch, frame]`.</param>
            /// <returns>
            /// The logits of masked frames. Tensor of dimension `[masked_frame, final_dim]`.
            /// The logits of unmasked frames. Tensor of dimension `[unmasked_frame, final_dim]`.
            /// </returns>
            public override (Tensor?, Tensor?) forward(Tensor x, Tensor label, Tensor mask_m, Tensor mask_u)
            {
                Tensor? logit_u;
                Tensor? logit_m;
                var proj_x = this.final_proj.call(x);
                if (this.skip_masked) {
                    logit_m = null;
                } else {
                    var proj_x_m = proj_x[mask_m];
                    var label_m = label[mask_m];
                    logit_m = _compute_logits(proj_x_m, label_m, this.label_embeddings);
                }

                if (this.skip_nomask) {
                    logit_u = null;
                } else {
                    var proj_x_u = proj_x[mask_u];
                    var label_u = label[mask_u];
                    logit_u = _compute_logits(proj_x_u, label_u, this.label_embeddings);
                }
                return (logit_m, logit_u);
            }
        }

#if true
        // TODO: https://github.com/dotnet/TorchSharp/issues/606
        internal static class GradMultiply
        {
            public static Tensor apply(Tensor x, double scale)
            {
                return x;
            }
        }
#else
        internal class GradMultiply : torch.autograd.Function
        {
            private double scale;

            public static Tensor forward(GradMultiply ctx, Tensor x, double scale)
            {
                ctx.scale = scale;
                var res = x.@new(x);
                return res;
            }

            public static (Tensor, object?) backward(GradMultiply ctx, Tensor grad)
            {
                return (grad * ctx.scale, null);
            }
        }
#endif
    }
}