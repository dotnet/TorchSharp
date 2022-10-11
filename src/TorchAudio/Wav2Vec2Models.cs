// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchaudio,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/audio/blob/76fca37ac8941b72a509a6e58d623632efe04543/torchaudio/models/wav2vec2/model.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/audio/blob/main/LICENSE
//

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

#nullable enable
namespace TorchSharp
{
    public static partial class torchaudio
    {
        public static partial class models
        {
            /// <summary>
            /// Build a custom Wav2Vec2Model
            /// 
            /// Note:
            /// The "feature extractor" below corresponds to
            /// `ConvFeatureExtractionModel` https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L736
            /// in the original ``fairseq`` implementation.
            /// This is referred as "(convolutional) feature encoder" in the *wav2vec 2.0*
            /// [:footcite:`baevski2020wav2vec`] paper.
            /// 
            /// The "encoder" below corresponds to `TransformerEncoder` https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L817,
            /// and this is referred as "Transformer" in the paper.
            /// </summary>
            /// <param name="extractor_mode">Operation mode of feature extractor.
            /// Valid values are ``"group_norm"`` or ``"layer_norm"``.
            /// If ``"group_norm"``, then a single normalization is applied
            /// in the first convolution block. Otherwise, all the convolution
            /// blocks will have layer normalization.
            /// 
            /// This option corresponds to ``extractor_mode`` from ``fairseq``.
            /// </param>
            /// <param name="extractor_conv_layer_config">
            /// Configuration of convolution layers in feature extractor.
            /// List of convolution configuration,
            /// i.e. ``[(output_channel, kernel_size, stride), ...]``
            /// 
            /// If ``None`` is provided, then the following default value is used.
            /// 
            /// .. code-block:: python
            /// 
            /// [
            /// (512, 10, 5),
            /// (512, 3, 2),
            /// (512, 3, 2),
            /// (512, 3, 2),
            /// (512, 3, 2),
            /// (512, 2, 2),
            /// (512, 2, 2),
            /// ]
            /// 
            /// This option corresponds to ``conv_feature_layers`` from ``fairseq``.
            /// </param>
            /// <param name="extractor_conv_bias">
            /// Whether to include bias term to each convolution operation.
            /// </param>
            /// This option corresponds to ``conv_bias`` from ``fairseq``.
            /// <param name="encoder_embed_dim">
            /// The dimension of embedding in encoder.
            /// 
            /// This option corresponds to ``encoder_embed_dim`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_projection_dropout">
            /// The dropout probability applied after the input feature is projected
            /// to ``encoder_embed_dim``.
            /// 
            /// This option corresponds to ``dropout_input`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_pos_conv_kernel">
            /// The kernel size of convolutional positional embeddings.
            /// 
            /// This option corresponds to ``conv_pos`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_pos_conv_groups">
            /// The number of groups of convolutional positional embeddings.
            /// 
            /// This option corresponds to ``conv_pos_groups`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_num_layers">
            /// The number of self attention layers in transformer block.
            /// 
            /// This option corresponds to ``encoder_layers`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_num_heads">
            /// The number of heads in self attention layers.
            /// 
            /// This option corresponds to ``encoder_attention_heads`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_attention_dropout">
            /// The dropout probability applied after softmax in self-attention layer.
            /// 
            /// This option corresponds to ``attention_dropout`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_ff_interm_features">
            /// The dimension of hidden features in feed forward layer.
            /// 
            /// This option corresponds to ``encoder_ffn_embed_dim`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_ff_interm_dropout">
            /// The dropout probability applied in feedforward layer.
            /// 
            /// This option correspinds to ``activation_dropout`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_dropout">
            /// The dropout probability applied at the end of feed forward layer.
            /// 
            /// This option corresponds to ``dropout`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_layer_norm_first">
            /// Control the order of layer norm in transformer layer and each encoder layer.
            /// If True, in transformer layer, layer norm is applied before features are fed
            /// to encoder layers. In encoder layer, two layer norms are applied before and after
            /// self attention.
            /// If False, in transformer layer, layer norm is applied after features are fed
            /// to encoder layers. In encoder layer, two layer norms are applied after self
            /// attention, before and after feed forward.
            /// 
            /// This option corresponds to ``layer_norm_first`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_layer_drop">
            /// Probability to drop each encoder layer during training.
            /// 
            /// This option corresponds to ``layerdrop`` from ``fairseq``.
            /// </param>
            /// <param name="aux_num_out">
            /// When provided, attach an extra linear layer on top of encoder, which can be
            /// used for fine-tuning.
            /// </param>
            /// <returns>
            /// The resulting model.
            /// </returns>
            public static Wav2Vec2Model wav2vec2_model(
                FeatureExtractorNormMode extractor_mode,
                long[][]? extractor_conv_layer_config,
                bool extractor_conv_bias,
                int encoder_embed_dim,
                double encoder_projection_dropout,
                int encoder_pos_conv_kernel,
                int encoder_pos_conv_groups,
                int encoder_num_layers,
                int encoder_num_heads,
                double encoder_attention_dropout,
                int encoder_ff_interm_features,
                double encoder_ff_interm_dropout,
                double encoder_dropout,
                bool encoder_layer_norm_first,
                double encoder_layer_drop,
                long? aux_num_out)
            {
                // Overriding the signature so that the return type is correct on Sphinx
                if (extractor_conv_layer_config == null) {
                    extractor_conv_layer_config = new long[][] {
                        new long[] { 512, 10, 5 },

                        new long[] { 512, 3, 2 },
                        new long[] { 512, 3, 2 },
                        new long[] { 512, 3, 2 },
                        new long[] { 512, 3, 2 },

                        new long[] { 512, 2, 2 },
                        new long[] { 512, 2, 2 },
                    };
                }

                var feature_extractor = Wav2Vec2Model._get_feature_extractor(
                    extractor_mode, extractor_conv_layer_config, extractor_conv_bias);

                var encoder = Wav2Vec2Model._get_encoder(
                    in_features: extractor_conv_layer_config[extractor_conv_layer_config.Length - 1][0],
                    embed_dim: encoder_embed_dim,
                    dropout_input: encoder_projection_dropout,
                    pos_conv_kernel: encoder_pos_conv_kernel,
                    pos_conv_groups: encoder_pos_conv_groups,
                    num_layers: encoder_num_layers,
                    num_heads: encoder_num_heads,
                    attention_dropout: encoder_attention_dropout,
                    ff_interm_features: encoder_ff_interm_features,
                    ff_interm_dropout: encoder_ff_interm_dropout,
                    dropout: encoder_dropout,
                    layer_norm_first: encoder_layer_norm_first,
                    layer_drop: encoder_layer_drop);
                Module<Tensor, Tensor>? aux = null;
                if (aux_num_out != null) {
                    aux = torch.nn.Linear(inputSize: encoder_embed_dim, outputSize: aux_num_out.Value);
                }
                return new Wav2Vec2Model("Wav2Vec2Model", feature_extractor, encoder, aux);
            }

            /// <summary>
            /// Build Wav2Vec2Model with "base" architecture from *wav2vec 2.0* [:footcite:`baevski2020wav2vec`]
            /// </summary>
            /// <param name="encoder_projection_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_attention_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_ff_interm_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_layer_drop">See `wav2vec2_model`.</param>
            /// <param name="aux_num_out">See `wav2vec2_model`.</param>
            /// <returns>
            /// The resulting model.
            /// </returns>
            public static Wav2Vec2Model wav2vec2_base(
                double encoder_projection_dropout = 0.1,
                double encoder_attention_dropout = 0.1,
                double encoder_ff_interm_dropout = 0.1,
                double encoder_dropout = 0.1,
                double encoder_layer_drop = 0.1,
                long? aux_num_out = null)
            {
                // Overriding the signature so that the return type is correct on Sphinx
                return wav2vec2_model(
                    extractor_mode: FeatureExtractorNormMode.group_norm,
                    extractor_conv_layer_config: null,
                    extractor_conv_bias: false,
                    encoder_embed_dim: 768,
                    encoder_projection_dropout: encoder_projection_dropout,
                    encoder_pos_conv_kernel: 128,
                    encoder_pos_conv_groups: 16,
                    encoder_num_layers: 12,
                    encoder_num_heads: 12,
                    encoder_attention_dropout: encoder_attention_dropout,
                    encoder_ff_interm_features: 3072,
                    encoder_ff_interm_dropout: encoder_ff_interm_dropout,
                    encoder_dropout: encoder_dropout,
                    encoder_layer_norm_first: false,
                    encoder_layer_drop: encoder_layer_drop,
                    aux_num_out: aux_num_out);
            }

            /// <summary>
            /// Build Wav2Vec2Model with "large" architecture from *wav2vec 2.0* [:footcite:`baevski2020wav2vec`]
            /// </summary>
            /// <param name="encoder_projection_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_attention_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_ff_interm_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_layer_drop">See `wav2vec2_model`.</param>
            /// <param name="aux_num_out">See `wav2vec2_model`.</param>
            /// <returns>
            /// The resulting model.
            /// </returns>
            public static Wav2Vec2Model wav2vec2_large(
                double encoder_projection_dropout = 0.1,
                double encoder_attention_dropout = 0.1,
                double encoder_ff_interm_dropout = 0.1,
                double encoder_dropout = 0.1,
                double encoder_layer_drop = 0.1,
                long? aux_num_out = null)
            {
                // Overriding the signature so that the return type is correct on Sphinx
                return wav2vec2_model(
                    extractor_mode: FeatureExtractorNormMode.group_norm,
                    extractor_conv_layer_config: null,
                    extractor_conv_bias: false,
                    encoder_embed_dim: 1024,
                    encoder_projection_dropout: encoder_projection_dropout,
                    encoder_pos_conv_kernel: 128,
                    encoder_pos_conv_groups: 16,
                    encoder_num_layers: 24,
                    encoder_num_heads: 16,
                    encoder_attention_dropout: encoder_attention_dropout,
                    encoder_ff_interm_features: 4096,
                    encoder_ff_interm_dropout: encoder_ff_interm_dropout,
                    encoder_dropout: encoder_dropout,
                    encoder_layer_norm_first: false,
                    encoder_layer_drop: encoder_layer_drop,
                    aux_num_out: aux_num_out);
            }

            /// <summary>
            /// Build Wav2Vec2Model with "large lv-60k" architecture from *wav2vec 2.0* [:footcite:`baevski2020wav2vec`]
            /// </summary>
            /// <param name="encoder_projection_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_attention_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_ff_interm_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_layer_drop">See `wav2vec2_model`.</param>
            /// <param name="aux_num_out">See `wav2vec2_model`.</param>
            /// <returns>
            /// The resulting model.
            /// </returns>
            public static Wav2Vec2Model wav2vec2_large_lv60k(
                double encoder_projection_dropout = 0.1,
                double encoder_attention_dropout = 0.0,
                double encoder_ff_interm_dropout = 0.1,
                double encoder_dropout = 0.0,
                double encoder_layer_drop = 0.1,
                long? aux_num_out = null)
            {
                // Overriding the signature so that the return type is correct on Sphinx
                return wav2vec2_model(
                    extractor_mode: FeatureExtractorNormMode.layer_norm,
                    extractor_conv_layer_config: null,
                    extractor_conv_bias: true,
                    encoder_embed_dim: 1024,
                    encoder_projection_dropout: encoder_projection_dropout,
                    encoder_pos_conv_kernel: 128,
                    encoder_pos_conv_groups: 16,
                    encoder_num_layers: 24,
                    encoder_num_heads: 16,
                    encoder_attention_dropout: encoder_attention_dropout,
                    encoder_ff_interm_features: 4096,
                    encoder_ff_interm_dropout: encoder_ff_interm_dropout,
                    encoder_dropout: encoder_dropout,
                    encoder_layer_norm_first: true,
                    encoder_layer_drop: encoder_layer_drop,
                    aux_num_out: aux_num_out);
            }

            /// <summary>
            /// Build HuBERT model with "base" architecture from *HuBERT* [:footcite:`hsu2021hubert`]
            /// </summary>
            /// <param name="encoder_projection_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_attention_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_ff_interm_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_layer_drop">See `wav2vec2_model`.</param>
            /// <param name="aux_num_out">See `wav2vec2_model`.</param>
            /// <returns>
            /// The resulting model.
            /// </returns>
            public static Wav2Vec2Model hubert_base(
                double encoder_projection_dropout = 0.1,
                double encoder_attention_dropout = 0.1,
                double encoder_ff_interm_dropout = 0.0,
                double encoder_dropout = 0.1,
                double encoder_layer_drop = 0.05,
                long? aux_num_out = null)
            {
                // Overriding the signature so that the return type is correct on Sphinx
                return wav2vec2_model(
                    extractor_mode: FeatureExtractorNormMode.group_norm,
                    extractor_conv_layer_config: null,
                    extractor_conv_bias: false,
                    encoder_embed_dim: 768,
                    encoder_projection_dropout: encoder_projection_dropout,
                    encoder_pos_conv_kernel: 128,
                    encoder_pos_conv_groups: 16,
                    encoder_num_layers: 12,
                    encoder_num_heads: 12,
                    encoder_attention_dropout: encoder_attention_dropout,
                    encoder_ff_interm_features: 3072,
                    encoder_ff_interm_dropout: encoder_ff_interm_dropout,
                    encoder_dropout: encoder_dropout,
                    encoder_layer_norm_first: false,
                    encoder_layer_drop: encoder_layer_drop,
                    aux_num_out: aux_num_out);
            }

            /// <summary>
            /// Build HuBERT model with "large" architecture from *HuBERT* [:footcite:`hsu2021hubert`]
            /// </summary>
            /// <param name="encoder_projection_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_attention_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_ff_interm_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_layer_drop">See `wav2vec2_model`.</param>
            /// <param name="aux_num_out">See `wav2vec2_model`.</param>
            /// <returns>
            /// The resulting model.
            /// </returns>
            public static Wav2Vec2Model hubert_large(
                double encoder_projection_dropout = 0.0,
                double encoder_attention_dropout = 0.0,
                double encoder_ff_interm_dropout = 0.0,
                double encoder_dropout = 0.0,
                double encoder_layer_drop = 0.0,
                long? aux_num_out = null)
            {
                // Overriding the signature so that the return type is correct on Sphinx
                return wav2vec2_model(
                    extractor_mode: FeatureExtractorNormMode.layer_norm,
                    extractor_conv_layer_config: null,
                    extractor_conv_bias: false,
                    encoder_embed_dim: 1024,
                    encoder_projection_dropout: encoder_projection_dropout,
                    encoder_pos_conv_kernel: 128,
                    encoder_pos_conv_groups: 16,
                    encoder_num_layers: 24,
                    encoder_num_heads: 16,
                    encoder_attention_dropout: encoder_attention_dropout,
                    encoder_ff_interm_features: 4096,
                    encoder_ff_interm_dropout: encoder_ff_interm_dropout,
                    encoder_dropout: encoder_dropout,
                    encoder_layer_norm_first: true,
                    encoder_layer_drop: encoder_layer_drop,
                    aux_num_out: aux_num_out);
            }

            /// <summary>
            /// Build HuBERT model with "extra large" architecture from *HuBERT* [:footcite:`hsu2021hubert`]
            /// </summary>
            /// <param name="encoder_projection_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_attention_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_ff_interm_dropout">See `wav2vec2_model`.</param>
            /// <param name="encoder_layer_drop">See `wav2vec2_model`.</param>
            /// <param name="aux_num_out">See `wav2vec2_model`.</param>
            /// <returns>
            /// The resulting model.
            /// </returns>
            public static Wav2Vec2Model hubert_xlarge(
                double encoder_projection_dropout = 0.0,
                double encoder_attention_dropout = 0.0,
                double encoder_ff_interm_dropout = 0.0,
                double encoder_dropout = 0.0,
                double encoder_layer_drop = 0.0,
                long? aux_num_out = null)
            {
                // Overriding the signature so that the return type is correct on Sphinx
                return wav2vec2_model(
                    extractor_mode: FeatureExtractorNormMode.layer_norm,
                    extractor_conv_layer_config: null,
                    extractor_conv_bias: false,
                    encoder_embed_dim: 1280,
                    encoder_projection_dropout: encoder_projection_dropout,
                    encoder_pos_conv_kernel: 128,
                    encoder_pos_conv_groups: 16,
                    encoder_num_layers: 48,
                    encoder_num_heads: 16,
                    encoder_attention_dropout: encoder_attention_dropout,
                    encoder_ff_interm_features: 5120,
                    encoder_ff_interm_dropout: encoder_ff_interm_dropout,
                    encoder_dropout: encoder_dropout,
                    encoder_layer_norm_first: true,
                    encoder_layer_drop: encoder_layer_drop,
                    aux_num_out: aux_num_out);
            }

            /// <summary>
            /// Build a custom HuBERTPretrainModel for training from scratch
            /// 
            /// Note:
            /// The "feature extractor" below corresponds to
            /// `ConvFeatureExtractionModel` https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L736
            /// in the original ``fairseq`` implementation.
            /// This is referred as "(convolutional) feature encoder" in the *wav2vec 2.0*
            /// [:footcite:`baevski2020wav2vec`] paper.
            /// 
            /// The "encoder" below corresponds to `TransformerEncoder` https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L817,
            /// and this is referred as "Transformer" in the paper.
            /// </summary>
            /// <param name="extractor_mode">
            /// Operation mode of feature extractor.
            /// Valid values are ``"group_norm"`` or ``"layer_norm"``.
            /// If ``"group_norm"``, then a single normalization is applied
            /// in the first convolution block. Otherwise, all the convolution
            /// blocks will have layer normalization.
            /// 
            /// This option corresponds to ``extractor_mode`` from ``fairseq``.
            /// </param>
            /// <param name="extractor_conv_layer_config">
            /// Configuration of convolution layers in feature extractor.
            /// List of convolution configuration,
            /// i.e. ``[(output_channel, kernel_size, stride), ...]``
            /// 
            /// If ``None`` is provided, then the following default value is used.
            /// 
            /// .. code-block:: python
            /// 
            /// [
            /// (512, 10, 5),
            /// (512, 3, 2),
            /// (512, 3, 2),
            /// (512, 3, 2),
            /// (512, 3, 2),
            /// (512, 2, 2),
            /// (512, 2, 2),
            /// ]
            /// 
            /// This option corresponds to ``conv_feature_layers`` from ``fairseq``.
            /// </param>
            /// <param name="extractor_conv_bias">
            /// Whether to include bias term to each convolution operation.
            /// 
            /// This option corresponds to ``conv_bias`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_embed_dim">
            /// The dimension of embedding in encoder.
            /// 
            /// This option corresponds to ``encoder_embed_dim`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_projection_dropout">
            /// The dropout probability applied after the input feature is projected
            /// to ``encoder_embed_dim``.
            /// 
            /// This option corresponds to ``dropout_input`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_pos_conv_kernel">
            /// The kernel size of convolutional positional embeddings.
            /// 
            /// This option corresponds to ``conv_pos`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_pos_conv_groups">
            /// The number of groups of convolutional positional embeddings.
            /// 
            /// This option corresponds to ``conv_pos_groups`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_num_layers">
            /// The number of self attention layers in transformer block.
            /// 
            /// This option corresponds to ``encoder_layers`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_num_heads">
            /// The number of heads in self attention layers.
            /// 
            /// This option corresponds to ``encoder_attention_heads`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_attention_dropout">
            /// The dropout probability applied after softmax in self-attention layer.
            /// 
            /// This option corresponds to ``attention_dropout`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_ff_interm_features">
            /// The dimension of hidden features in feed forward layer.
            /// 
            /// This option corresponds to ``encoder_ffn_embed_dim`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_ff_interm_dropout">
            /// The dropout probability applied in feedforward layer.
            /// 
            /// This option correspinds to ``activation_dropout`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_dropout">
            /// The dropout probability applied at the end of feed forward layer.
            /// 
            /// This option corresponds to ``dropout`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_layer_norm_first">
            /// Control the order of layer norm in transformer layer and each encoder layer.
            /// If True, in transformer layer, layer norm is applied before features are fed
            /// to encoder layers. In encoder layer, two layer norms are applied before and after
            /// self attention.
            /// If False, in transformer layer, layer norm is applied after features are fed
            /// to encoder layers. In encoder layer, two layer norms are applied after self
            /// attention, before and after feed forward.
            /// 
            /// This option corresponds to ``layer_norm_first`` from ``fairseq``.
            /// </param>
            /// <param name="encoder_layer_drop">
            /// Probability to drop each encoder layer during training.
            /// 
            /// This option corresponds to ``layerdrop`` from ``fairseq``.
            /// </param>
            /// <param name="mask_prob">
            /// Probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            /// number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            /// However due to overlaps, the actual number will be smaller (unless no_overlap is True).
            /// 
            /// This option corresponds to ``mask_prob`` from ``fairseq``.
            /// </param>
            /// <param name="mask_selection">
            /// How to choose the mask length. Options: [``static``, ``uniform``, ``normal``, ``poisson``].
            /// 
            /// This option corresponds to ``mask_selection`` from ``fairseq``.
            /// </param>
            /// <param name="mask_other">
            /// Secondary mask argument (used for more complex distributions).
            /// 
            /// This option corresponds to ``mask_other`` from ``fairseq``.
            /// </param>
            /// <param name="mask_length">
            /// The lengths of the mask.
            /// 
            /// This option corresponds to ``mask_length`` from ``fairseq``.
            /// </param>
            /// <param name="no_mask_overlap">
            /// Whether to allow masks to overlap.
            /// 
            /// This option corresponds to ``no_mask_overlap`` from ``fairseq``.
            /// </param>
            /// <param name="mask_min_space">
            /// Minimum space between spans (if no overlap is enabled).
            /// 
            /// This option corresponds to ``mask_min_space`` from ``fairseq``.
            /// </param>
            /// <param name="mask_channel_prob">
            /// The probability of replacing a feature with 0.
            /// 
            /// This option corresponds to ``mask_channel_prob`` from ``fairseq``.
            /// </param>
            /// <param name="mask_channel_selection">
            /// How to choose the mask length for channel masking. Options: [``static``, ``uniform``, ``normal``, ``poisson``].
            /// 
            /// This option corresponds to ``mask_channel_selection`` from ``fairseq``.
            /// </param>
            /// <param name="mask_channel_other">
            /// Secondary mask argument for channel masking(used for more complex distributions).
            /// 
            /// This option corresponds to ``mask_channel_other`` from ``fairseq``.
            /// </param>
            /// <param name="mask_channel_length">
            /// Minimum space between spans (if no overlap is enabled) for channel masking.
            /// 
            /// This option corresponds to ``mask_channel_length`` from ``fairseq``.
            /// </param>
            /// <param name="no_mask_channel_overlap">
            /// Whether to allow channel masks to overlap.
            /// 
            /// This option corresponds to ``no_mask_channel_overlap`` from ``fairseq``.
            /// </param>
            /// <param name="mask_channel_min_space">
            /// Minimum space between spans for channel masking(if no overlap is enabled).
            /// 
            /// This option corresponds to ``mask_channel_min_space`` from ``fairseq``.
            /// </param>
            /// <param name="skip_masked">
            /// If True, skip computing losses over masked frames.
            /// 
            /// This option corresponds to ``skip_masked`` from ``fairseq``.
            /// </param>
            /// <param name="skip_nomask">
            /// If True, skip computing losses over unmasked frames.
            /// 
            /// This option corresponds to ``skip_nomask`` from ``fairseq``.
            /// </param>
            /// <param name="num_classes">
            /// The number of classes in the labels.
            /// </param>
            /// <param name="final_dim">
            /// Project final representations and targets to `final_dim`.
            /// 
            /// This option corresponds to ``final_dim`` from ``fairseq``.
            /// </param>
            /// <param name="feature_grad_mult">
            /// The factor to scale the convolutional feature extraction layer gradients by.
            /// The scale factor will not affect the forward pass.
            /// 
            /// This option corresponds to ``feature_grad_mult`` from ``fairseq``.
            /// </param>
            /// <returns>
            /// The resulting model.
            /// </returns>
            public static HuBERTPretrainModel hubert_pretrain_model(
                FeatureExtractorNormMode extractor_mode,
                long[][]? extractor_conv_layer_config,
                bool extractor_conv_bias,
                int encoder_embed_dim,
                double encoder_projection_dropout,
                int encoder_pos_conv_kernel,
                int encoder_pos_conv_groups,
                int encoder_num_layers,
                int encoder_num_heads,
                double encoder_attention_dropout,
                int encoder_ff_interm_features,
                double encoder_ff_interm_dropout,
                double encoder_dropout,
                bool encoder_layer_norm_first,
                double encoder_layer_drop,
                double mask_prob,
                string mask_selection,
                double mask_other,
                int mask_length,
                bool no_mask_overlap,
                int mask_min_space,
                double mask_channel_prob,
                string mask_channel_selection,
                double mask_channel_other,
                long mask_channel_length,
                bool no_mask_channel_overlap,
                int mask_channel_min_space,
                bool skip_masked,
                bool skip_nomask,
                long num_classes,
                int final_dim,
                double? feature_grad_mult)
            {
                // Overriding the signature so that the return type is correct on Sphinx
                if (extractor_conv_layer_config == null) {
                    extractor_conv_layer_config = new long[][] {
                        new long[] { 512, 10, 5 },

                        new long[] { 512, 3, 2 },
                        new long[] { 512, 3, 2 },
                        new long[] { 512, 3, 2 },
                        new long[] { 512, 3, 2 },

                        new long[] { 512, 2, 2 },
                        new long[] { 512, 2, 2 },
                    };
                }

                var feature_extractor = Wav2Vec2Model._get_feature_extractor(
                    extractor_mode, extractor_conv_layer_config, extractor_conv_bias);
                var encoder = Wav2Vec2Model._get_encoder(
                    in_features: extractor_conv_layer_config[extractor_conv_layer_config.Length - 1][0],
                    embed_dim: encoder_embed_dim,
                    dropout_input: encoder_projection_dropout,
                    pos_conv_kernel: encoder_pos_conv_kernel,
                    pos_conv_groups: encoder_pos_conv_groups,
                    num_layers: encoder_num_layers,
                    num_heads: encoder_num_heads,
                    attention_dropout: encoder_attention_dropout,
                    ff_interm_features: encoder_ff_interm_features,
                    ff_interm_dropout: encoder_ff_interm_dropout,
                    dropout: encoder_dropout,
                    layer_norm_first: encoder_layer_norm_first,
                    layer_drop: encoder_layer_drop);
                var wav2vec2 = new Wav2Vec2Model("Wav2Vec2Model", feature_extractor, encoder);
                var mask_generator = new Wav2Vec2Model.MaskGenerator(
                    "MaskGenerator",
                    encoder_embed_dim,
                    mask_prob,
                    mask_selection,
                    mask_other,
                    mask_length,
                    no_mask_overlap,
                    mask_min_space,
                    mask_channel_prob,
                    mask_channel_selection,
                    mask_channel_other,
                    mask_channel_length,
                    no_mask_channel_overlap,
                    mask_channel_min_space);
                var logit_generator = new Wav2Vec2Model.LogitGenerator(
                    "LogitGenerator",
                    encoder_embed_dim,
                    num_classes,
                    final_dim,
                    skip_masked,
                    skip_nomask);
                return new HuBERTPretrainModel(
                    "HuBERTPretrainModel",
                    wav2vec2: wav2vec2,
                    mask_generator: mask_generator,
                    logit_generator: logit_generator,
                    feature_grad_mult: feature_grad_mult);
            }

            /// <summary>
            /// Build HuBERTPretrainModel model with "base" architecture from *HuBERT* [:footcite:`hsu2021hubert`]
            /// </summary>
            /// <param name="encoder_projection_dropout">See `hubert_pretrain_model`.</param>
            /// <param name="encoder_attention_dropout">See `hubert_pretrain_model`.</param>
            /// <param name="encoder_dropout">See `hubert_pretrain_model`.</param>
            /// <param name="encoder_ff_interm_dropout">See `hubert_pretrain_model`.</param>
            /// <param name="encoder_layer_drop">See `hubert_pretrain_model`.</param>
            /// <param name="mask_prob">See `hubert_pretrain_model`.</param>
            /// <param name="mask_channel_prob">See `hubert_pretrain_model`.</param>
            /// <param name="mask_channel_length">See `hubert_pretrain_model`.</param>
            /// <param name="feature_grad_mult">See `hubert_pretrain_model`.</param>
            /// <param name="num_classes">See `hubert_pretrain_model`.</param>
            /// <returns>
            /// The resulting model.
            /// </returns>
            public static HuBERTPretrainModel hubert_pretrain_base(
                double encoder_projection_dropout = 0.1,
                double encoder_attention_dropout = 0.1,
                double encoder_ff_interm_dropout = 0.0,
                double encoder_dropout = 0.1,
                double encoder_layer_drop = 0.05,
                double mask_prob = 0.8,
                double mask_channel_prob = 0.0,
                long mask_channel_length = 10,
                double? feature_grad_mult = 0.1,
                long num_classes = 100)
            {
                // Overriding the signature so that the return type is correct on Sphinx
                return hubert_pretrain_model(
                    extractor_mode: FeatureExtractorNormMode.group_norm,
                    extractor_conv_layer_config: null,
                    extractor_conv_bias: false,
                    encoder_embed_dim: 768,
                    encoder_projection_dropout: encoder_projection_dropout,
                    encoder_pos_conv_kernel: 128,
                    encoder_pos_conv_groups: 16,
                    encoder_num_layers: 12,
                    encoder_num_heads: 12,
                    encoder_attention_dropout: encoder_attention_dropout,
                    encoder_ff_interm_features: 3072,
                    encoder_ff_interm_dropout: encoder_ff_interm_dropout,
                    encoder_dropout: encoder_dropout,
                    encoder_layer_norm_first: false,
                    encoder_layer_drop: encoder_layer_drop,
                    mask_prob: mask_prob,
                    mask_selection: "static",
                    mask_other: 0.0,
                    mask_length: 10,
                    no_mask_overlap: false,
                    mask_min_space: 1,
                    mask_channel_prob: mask_channel_prob,
                    mask_channel_selection: "static",
                    mask_channel_other: 0.0,
                    mask_channel_length: mask_channel_length,
                    no_mask_channel_overlap: false,
                    mask_channel_min_space: 1,
                    skip_masked: false,
                    skip_nomask: false,
                    num_classes: num_classes,
                    final_dim: 256,
                    feature_grad_mult: feature_grad_mult);
            }

            /// <summary>
            /// Build HuBERTPretrainModel model for pre-training with "large" architecture from *HuBERT* [:footcite:`hsu2021hubert`]
            /// </summary>
            /// <param name="encoder_projection_dropout">See `hubert_pretrain_model`.</param>
            /// <param name="encoder_attention_dropout">See `hubert_pretrain_model`.</param>
            /// <param name="encoder_dropout">See `hubert_pretrain_model`.</param>
            /// <param name="encoder_ff_interm_dropout">See `hubert_pretrain_model`.</param>
            /// <param name="encoder_layer_drop">See `hubert_pretrain_model`.</param>
            /// <param name="mask_prob">See `hubert_pretrain_model`.</param>
            /// <param name="mask_channel_prob">See `hubert_pretrain_model`.</param>
            /// <param name="mask_channel_length">See `hubert_pretrain_model`.</param>
            /// <param name="feature_grad_mult">See `hubert_pretrain_model`.</param>
            /// <returns>
            /// The resulting model.
            /// </returns>
            public static HuBERTPretrainModel hubert_pretrain_large(
                double encoder_projection_dropout = 0.0,
                double encoder_attention_dropout = 0.0,
                double encoder_ff_interm_dropout = 0.0,
                double encoder_dropout = 0.0,
                double encoder_layer_drop = 0.0,
                double mask_prob = 0.8,
                double mask_channel_prob = 0.0,
                long mask_channel_length = 10,
                double? feature_grad_mult = null)
            {
                // Overriding the signature so that the return type is correct on Sphinx
                return hubert_pretrain_model(
                    extractor_mode: FeatureExtractorNormMode.layer_norm,
                    extractor_conv_layer_config: null,
                    extractor_conv_bias: false,
                    encoder_embed_dim: 1024,
                    encoder_projection_dropout: encoder_projection_dropout,
                    encoder_pos_conv_kernel: 128,
                    encoder_pos_conv_groups: 16,
                    encoder_num_layers: 24,
                    encoder_num_heads: 16,
                    encoder_attention_dropout: encoder_attention_dropout,
                    encoder_ff_interm_features: 4096,
                    encoder_ff_interm_dropout: encoder_ff_interm_dropout,
                    encoder_dropout: encoder_dropout,
                    encoder_layer_norm_first: true,
                    encoder_layer_drop: encoder_layer_drop,
                    mask_prob: mask_prob,
                    mask_selection: "static",
                    mask_other: 0.0,
                    mask_length: 10,
                    no_mask_overlap: false,
                    mask_min_space: 1,
                    mask_channel_prob: mask_channel_prob,
                    mask_channel_selection: "static",
                    mask_channel_other: 0.0,
                    mask_channel_length: mask_channel_length,
                    no_mask_channel_overlap: false,
                    mask_channel_min_space: 1,
                    skip_masked: false,
                    skip_nomask: false,
                    num_classes: 500,
                    final_dim: 768,
                    feature_grad_mult: feature_grad_mult);
            }

            /// <summary>
            /// Build HuBERTPretrainModel model for pre-training with "extra large" architecture from *HuBERT* [:footcite:`hsu2021hubert`]
            /// </summary>
            /// <param name="encoder_projection_dropout">See `hubert_pretrain_model`.</param>
            /// <param name="encoder_attention_dropout">See `hubert_pretrain_model`.</param>
            /// <param name="encoder_dropout">See `hubert_pretrain_model`.</param>
            /// <param name="encoder_ff_interm_dropout">See `hubert_pretrain_model`.</param>
            /// <param name="encoder_layer_drop">See `hubert_pretrain_model`.</param>
            /// <param name="mask_prob">See `hubert_pretrain_model`.</param>
            /// <param name="mask_channel_prob">See `hubert_pretrain_model`.</param>
            /// <param name="mask_channel_length">See `hubert_pretrain_model`.</param>
            /// <param name="feature_grad_mult">See `hubert_pretrain_model`.</param>
            /// <returns>
            /// The resulting model.
            /// </returns>
            public static HuBERTPretrainModel hubert_pretrain_xlarge(
                double encoder_projection_dropout = 0.0,
                double encoder_attention_dropout = 0.0,
                double encoder_ff_interm_dropout = 0.0,
                double encoder_dropout = 0.0,
                double encoder_layer_drop = 0.0,
                double mask_prob = 0.8,
                double mask_channel_prob = 0.0,
                long mask_channel_length = 10,
                double? feature_grad_mult = null)
            {
                // Overriding the signature so that the return type is correct on Sphinx
                return hubert_pretrain_model(
                    extractor_mode: FeatureExtractorNormMode.layer_norm,
                    extractor_conv_layer_config: null,
                    extractor_conv_bias: false,
                    encoder_embed_dim: 1280,
                    encoder_projection_dropout: encoder_projection_dropout,
                    encoder_pos_conv_kernel: 128,
                    encoder_pos_conv_groups: 16,
                    encoder_num_layers: 48,
                    encoder_num_heads: 16,
                    encoder_attention_dropout: encoder_attention_dropout,
                    encoder_ff_interm_features: 5120,
                    encoder_ff_interm_dropout: encoder_ff_interm_dropout,
                    encoder_dropout: encoder_dropout,
                    encoder_layer_norm_first: true,
                    encoder_layer_drop: encoder_layer_drop,
                    mask_prob: mask_prob,
                    mask_selection: "static",
                    mask_other: 0.0,
                    mask_length: 10,
                    no_mask_overlap: false,
                    mask_min_space: 1,
                    mask_channel_prob: mask_channel_prob,
                    mask_channel_selection: "static",
                    mask_channel_other: 0.0,
                    mask_channel_length: mask_channel_length,
                    no_mask_channel_overlap: false,
                    mask_channel_min_space: 1,
                    skip_masked: false,
                    skip_nomask: false,
                    num_classes: 500,
                    final_dim: 1024,
                    feature_grad_mult: feature_grad_mult);
            }
        }
    }
}
