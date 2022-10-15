// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

namespace TorchSharp
{
    public static partial class torchaudio
    {
        /// <summary>
        /// Load audio data from a file.
        /// </summary>
        /// <param name="filepath">The file path to load</param>
        /// <param name="frame_offset">Number of frames to skip before reading data</param>
        /// <param name="num_frames">Maximum number of frames to read.</param>
        /// <param name="normalize">True to normalize the audio to [-1.0, 1.0]</param>
        /// <param name="channels_first">The dimension of the returned tensor is [channel, time] when true</param>
        /// <param name="format">The format of the audio file</param>
        /// <returns>A pair of waveform and sampling rate</returns>
        public static (torch.Tensor, int) load(
            string filepath,
            long frame_offset = 0,
            long num_frames = -1,
            bool normalize = true,
            bool channels_first = true,
            AudioFormat? format = null)
        {
            return backend.utils._backend.load(
                filepath,
                frame_offset,
                num_frames,
                normalize,
                channels_first,
                format);
        }

        /// <summary>
        /// Save audio data to a file.
        /// </summary>
        /// <param name="filepath">The file path to save</param>
        /// <param name="src">The waveform of audio</param>
        /// <param name="sample_rate">The sampling rate of audio</param>
        /// <param name="channels_first">The dimension of the returned tensor is [channel, time] when true</param>
        /// <param name="compression">The compression factor</param>
        /// <param name="format">The format of the audio file</param>
        /// <param name="encoding">The audio encoding</param>
        /// <param name="bits_per_sample">The number of bits per sample</param>
        public static void save(
            string filepath,
            torch.Tensor src,
            int sample_rate,
            bool channels_first = true,
            float? compression = null,
            AudioFormat? format = null,
            AudioEncoding? encoding = null,
            int? bits_per_sample = null)
        {
            backend.utils._backend.save(
                filepath,
                src,
                sample_rate,
                channels_first,
                compression,
                format,
                encoding,
                bits_per_sample);
        }

        /// <summary>
        /// Get the information of an audio file.
        /// </summary>
        /// <param name="filepath">The file path of the audio file</param>
        /// <param name="format">The format of the audio file</param>
        /// <returns>The information of the audio file</returns>
        public static AudioMetaData info(
            string filepath,
            AudioFormat? format = null)
        {
            return backend.utils._backend.info(
                filepath,
                format);
        }
    }
}
