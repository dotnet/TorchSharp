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

#nullable enable
namespace TorchSharp.Modules
{
    /// <summary>
    /// Encoder model used in *wav2vec 2.0* [:footcite:`baevski2020wav2vec`].
    /// Note:
    /// To build the model, please use one of the factory functions.
    /// </summary>
    public partial class Wav2Vec2Model : nn.Module<Tensor, Tensor?, (Tensor, Tensor?)>
    {
        internal readonly FeatureExtractor feature_extractor;
        internal readonly Encoder encoder;
        private readonly nn.Module<Tensor, Tensor>? aux;

        /// <param name="name"></param>
        /// <param name="feature_extractor">Feature extractor that extracts feature vectors from raw audio Tensor.</param>
        /// <param name="encoder">Encoder that converts the audio features into the sequence of probability
        /// distribution (in negative log-likelihood) over labels.</param>
        /// <param name="aux">Auxiliary module. If provided, the output from encoder is passed to this module.</param>
        internal Wav2Vec2Model(
            string name,
            FeatureExtractor feature_extractor,
            Encoder encoder,
            nn.Module<Tensor, Tensor>? aux = null) : base(name)
        {
            this.feature_extractor = feature_extractor;
            this.encoder = encoder;
            this.aux = aux;
            RegisterComponents();
        }

        /// <summary>
        /// Extract feature vectors from raw waveforms
        /// 
        /// This returns the list of outputs from the intermediate layers of
        /// transformer block in encoder.
        /// </summary>
        /// <param name="waveforms">Audio tensor of shape `(batch, frames)`.</param>
        /// <param name="lengths">Indicates the valid length of each audio in the batch.
        /// Shape: `(batch, )`.
        /// When the ``waveforms`` contains audios with different durations,
        /// by providing ``lengths`` argument, the model will compute
        /// the corresponding valid output lengths and apply proper mask in
        /// transformer attention layer.
        /// If ``None``, it is assumed that the entire audio waveform
        /// length is valid.</param>
        /// <param name="num_layers">
        /// If given, limit the number of intermediate layers to go through.
        /// Providing `1` will stop the computation after going through one
        /// intermediate layers. If not given, the outputs from all the
        /// intermediate layers are returned.
        /// </param>
        /// <returns>
        /// List of Tensors
        /// Features from requested layers.
        /// Each Tensor is of shape: `(batch, time frame, feature dimension)`
        /// If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
        /// is returned.
        /// It indicates the valid length in time axis of each feature Tensor.
        /// </returns>
        public (Tensor[], Tensor?) extract_features(
            Tensor waveforms,
            Tensor? lengths = null,
            int? num_layers = null)
        {
            Tensor x;
            (x, lengths) = this.feature_extractor.call(waveforms, lengths);
            var xs = this.encoder.extract_features(x, lengths, num_layers);
            return (xs, lengths);
        }

        /// <summary>
        /// Compute the sequence of probability distribution over labels.
        /// </summary>
        /// <param name="waveforms">Audio tensor of shape `(batch, frames)`.
        /// waveforms (Tensor): Audio tensor of shape `(batch, frames)`.</param>
        /// <param name="lengths">Indicates the valid length of each audio in the batch.
        /// Shape: `(batch, )`.
        /// When the ``waveforms`` contains audios with different durations,
        /// by providing ``lengths`` argument, the model will compute
        /// the corresponding valid output lengths and apply proper mask in
        /// transformer attention layer.
        /// If ``None``, it is assumed that all the audio in ``waveforms``
        /// have valid length. Default: ``None``.</param>
        /// <returns>
        /// The sequences of probability distribution (in logit) over labels.
        /// If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
        /// is returned.
        /// It indicates the valid length in time axis of the output Tensor.
        /// </returns>
        public override (Tensor, Tensor?) forward(
            Tensor waveforms,
            Tensor? lengths = null)
        {
            Tensor x;
            (x, lengths) = this.feature_extractor.call(waveforms, lengths);
            x = this.encoder.call(x, lengths);
            if (this.aux != null) {
                x = this.aux.call(x);
            }
            return (x, lengths);
        }

        public new (Tensor, Tensor?) call(Tensor waveforms, Tensor? lengths = null) => base.call(waveforms, lengths);
    }
}
