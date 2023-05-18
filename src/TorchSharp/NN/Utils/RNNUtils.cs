// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class nn
        {
            public static partial class utils
            {
                public static partial class rnn
                {
                    /// <summary>
                    /// Pack a padded sequences.
                    /// </summary>
                    /// <param name="input">The padded input tensor</param>
                    /// <param name="lengths">The lengths of sequences</param>
                    /// <param name="batch_first">If true, the size of input tensor is (batch, seq, *), otherwise (seq, batch, *)</param>
                    /// <param name="enforce_sorted">if true, checks that the input contains sequences sorted by length in a decreasing order.</param>
                    /// <returns></returns>
                    public static PackedSequence pack_padded_sequence(torch.Tensor input, torch.Tensor lengths, bool batch_first = false, bool enforce_sorted = true)
                    {
                        var res = THSNN_pack_padded_sequence(input.Handle, lengths.Handle, batch_first, enforce_sorted);
                        if (res.IsInvalid) { torch.CheckForErrors(); }
                        return new PackedSequence(res);
                    }

                    /// <summary>
                    /// Pad a packed batch of variable length sequences.
                    /// </summary>
                    /// <param name="sequence">A packed batch</param>
                    /// <param name="batch_first">If true, the size of output tensor is (batch, seq, *), otherwise (seq, batch, *)</param>
                    /// <param name="padding_value">The values of padding.</param>
                    /// <param name="total_length">If not null, the output tensor is padded to the length of total_length.</param>
                    /// <returns>The padded tensor</returns>
                    public static (torch.Tensor, torch.Tensor) pad_packed_sequence(PackedSequence sequence, bool batch_first = false, double padding_value = 0.0, long? total_length = null)
                    {
                        IntPtr res1, res2;
                        long total_length_arg = total_length.HasValue ? total_length.Value : -1;
                        THSNN_pad_packed_sequence(sequence.Handle, batch_first, padding_value, total_length_arg, out res1, out res2);
                        if (res1 == IntPtr.Zero || res2 == IntPtr.Zero) { torch.CheckForErrors(); }
                        return (new torch.Tensor(res1), new torch.Tensor(res2));
                    }

                    /// <summary>
                    /// Pad a list of variable length sequences.
                    /// </summary>
                    /// <param name="sequences">A list of variable length sequences</param>
                    /// <param name="batch_first">If true, the size of output tensor is (batch, seq, *), otherwise (seq, batch, *)</param>
                    /// <param name="padding_value">The values of padding.</param>
                    /// <returns>The padded tensor</returns>
                    public static torch.Tensor pad_sequence(IEnumerable<torch.Tensor> sequences, bool batch_first = false, double padding_value = 0.0)
                    {
                        var sequences_arg = sequences.Select(p => p.Handle).ToArray();
                        var res = THSNN_pad_sequence(sequences_arg, sequences_arg.Length, batch_first, padding_value);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new torch.Tensor(res);
                    }

                    /// <summary>
                    /// Pack a list of variable length sequences.
                    /// </summary>
                    /// <param name="sequences">A list of variable length sequences</param>
                    /// <param name="enforce_sorted">if true, checks that the input contains sequences sorted by length in a decreasing order.</param>
                    /// <returns>The packed batch of variable length sequences</returns>
                    public static PackedSequence pack_sequence(IEnumerable<torch.Tensor> sequences, bool enforce_sorted = true)
                    {
                        var sequences_arg = sequences.Select(p => p.Handle).ToArray();
                        var res = THSNN_pack_sequence(sequences_arg, sequences_arg.Length, enforce_sorted);
                        if (res.IsInvalid) { torch.CheckForErrors(); }
                        return new PackedSequence(res);
                    }
                }
            }
        }
    }
}
