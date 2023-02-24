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

using System;
using System.IO;

using static TorchSharp.torch;

#nullable enable
namespace TorchSharp.Modules
{
    /// <summary>
    /// HuBERT pre-train model for training from scratch.
    /// 
    /// Note:
    /// To build the model, please use one of the factory functions in
    /// `[hubert_pretrain_base, hubert_pretrain_large, hubert_pretrain_xlarge]`.
    /// </summary>
    public class HuBERTPretrainModel : nn.Module<Tensor, Tensor, Tensor?, (Tensor?, Tensor?, Tensor)>
    {
        private readonly Wav2Vec2Model wav2vec2;
        private readonly Wav2Vec2Model.MaskGenerator mask_generator;
        private readonly Wav2Vec2Model.LogitGenerator logit_generator;
        private readonly double? feature_grad_mult;

        /// <param name="name"></param>
        /// <param name="wav2vec2"></param>
        /// <param name="mask_generator">Mask generator that generates the mask for masked prediction during the training.</param>
        /// <param name="logit_generator">Logit generator that predicts the logits of the masked and unmasked inputs.</param>
        /// <param name="feature_grad_mult">The factor to scale the convolutional feature extraction layer gradients by.
        /// If ``None``, the gradients of feature extraction layers are not affected.
        /// The scale factor will not affect the forward pass.</param>
        internal HuBERTPretrainModel(
            string name,
            Wav2Vec2Model wav2vec2,
            Wav2Vec2Model.MaskGenerator mask_generator,
            Wav2Vec2Model.LogitGenerator logit_generator,
            double? feature_grad_mult) : base(name)
        {
            this.wav2vec2 = wav2vec2;
            this.mask_generator = mask_generator;
            this.logit_generator = logit_generator;
            if (feature_grad_mult != null && !(0.0 < feature_grad_mult.Value && feature_grad_mult.Value < 1.0)) {
                throw new ArgumentException(
                    $"The value of `feature_grad_mult` must be ``null``or between (0, 1). Found {feature_grad_mult}");
            }
            this.feature_grad_mult = feature_grad_mult;
            RegisterComponents();
        }

        /// <summary>
        /// Compute the sequence of probability distribution over labels.
        /// </summary>
        /// <param name="waveforms">Audio tensor of dimension `[batch, frames]`.</param>
        /// <param name="labels">Label for pre-training. A Tensor of dimension `[batch, frames]`.</param>
        /// <param name="audio_lengths">Indicates the valid length of each audio in the batch.
        /// Shape: `[batch, ]`.
        /// When the ``waveforms`` contains audios with different durations,
        /// by providing ``lengths`` argument, the model will compute
        /// the corresponding valid output lengths and apply proper mask in
        /// transformer attention layer.
        /// If ``None``, it is assumed that all the audio in ``waveforms``
        /// have valid length. Default: ``None``.</param>
        /// <returns>
        /// The masked sequences of probability distribution (in logit).
        /// Shape: `(masked_frames, num labels)`.
        /// The unmasked sequence of probability distribution (in logit).
        /// Shape: `(unmasked_frames, num labels)`.
        /// Tensor
        /// The feature mean value for additional penalty loss.
        /// Shape: `(1,)`.
        /// </returns>
        public override (Tensor?, Tensor?, Tensor) forward(
            Tensor waveforms,
            Tensor labels,
            Tensor? audio_lengths = null)
        {
            Tensor mask_u;
            Tensor mask_m;
            Tensor? padding_mask;
            var (x, lengths) = this.wav2vec2.feature_extractor.call(waveforms, audio_lengths);
            if (this.feature_grad_mult != null && this.feature_grad_mult < 1.0) {
                x = Wav2Vec2Model.GradMultiply.apply(x, this.feature_grad_mult.Value);
            }
            var features_pen = x.@float().pow(2).mean();
            if (lengths is not null) {
                padding_mask = Wav2Vec2Model._get_padding_mask(x, lengths);
            } else {
                padding_mask = null;
            }
            Tensor? attention_mask;
            Tensor? mask;
            (x, attention_mask) = this.wav2vec2.encoder._preprocess(x, lengths);
            (x, mask) = this.mask_generator.call(x, padding_mask);
            if (mask is null) throw new InvalidDataException();
            x = this.wav2vec2.encoder.transformer.call(x, attention_mask: attention_mask);
            if (x.shape[1] != labels.shape[1]) {
                throw new ArgumentException("The length of label must match that of HuBERT model output");
            }
            if (padding_mask is not null) {
                mask_m = torch.logical_and(~padding_mask, mask);
                mask_u = torch.logical_and(~padding_mask, ~mask_m);
            } else {
                mask_m = mask;
                mask_u = ~mask_m;
            }

            var (logit_m, logit_u) = this.logit_generator.call(x, labels, mask_m, mask_u);

            return (logit_m, logit_u, features_pen);
        }
    }
}
