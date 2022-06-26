// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using Xunit;

using TorchSharp;

namespace TorchSharp
{
    public class TestNNUtils
    {
        (torch.Tensor[], torch.Tensor) make_test()
        {
            var sequences = new torch.Tensor[]
            {
                torch.tensor(new long[] { 1, 2, 3, 4 }),
                torch.tensor(new long[] { 5, 6 }),
                torch.tensor(new long[] { 7, 8 }),
            };
            var sequences_len = torch.tensor(sequences.Select(x => x.shape[0]).ToArray());
            return (sequences, sequences_len);
        }

        [Fact]
        public void TestPadSequence()
        {
            var (sequences, sequences_len) = make_test();
            var padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences);
            Assert.Equal(2, padded_sequences.ndim);
            Assert.Equal(new long[] { 4, 3 }, padded_sequences.shape);
            Assert.True(padded_sequences.sum().item<long>() == 36);

            var packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(padded_sequences, sequences_len);
            var (inverted_sequences, inverted_sequences_len) = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence);
            packed_sequence.Dispose();
            Assert.True(torch.max(torch.square(inverted_sequences - padded_sequences)).item<long>() == 0);
        }

        [Fact]
        public void TestPackSequence()
        {
            var (sequences, sequences_len) = make_test();
            var padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences);
            Assert.Equal(2, padded_sequences.ndim);
            Assert.Equal(new long[] { 4, 3 }, padded_sequences.shape);
            Assert.True(padded_sequences.sum().item<long>() == 36);

            var packed_sequence = torch.nn.utils.rnn.pack_sequence(sequences);
            var (inverted_sequences, inverted_sequences_len) = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence);
            packed_sequence.Dispose();
            Assert.True(torch.max(torch.square(inverted_sequences - padded_sequences)).item<long>() == 0);
        }
    }
}
