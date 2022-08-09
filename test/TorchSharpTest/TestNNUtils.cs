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

        [Fact]
        public void TestAutoGradGrad()
        {
            using var _ = torch.NewDisposeScope();
            var x1 = torch.rand(1, requiresGrad: true);
            var x2 = torch.rand(1, requiresGrad: true);

            var y = x1.pow(2) + 5 * x2;

            var grad = torch.autograd.grad(new[] { y }, new[] { x1, x2 }, new[] { torch.ones_like(y) });
            Assert.Equal(x1.shape, grad[0].shape);
            Assert.Equal(x2.shape, grad[1].shape);
            Assert.Equal(2.0f * x1.item<float>(), grad[0].item<float>());
            Assert.Equal(5.0f, grad[1].item<float>());
        }

        [Fact]
        public void TestAutoGradBackward1()
        {
            using var _ = torch.NewDisposeScope();
            var x1 = torch.rand(1, requiresGrad: true);
            var x2 = torch.rand(1, requiresGrad: true);

            var y = x1.pow(2) + 5 * x2;

            torch.autograd.backward(new[] { y }, new[] { torch.ones_like(y) });
            Assert.Equal(x1.shape, x1.grad().shape);
            Assert.Equal(x2.shape, x2.grad().shape);
            Assert.Equal(2.0f*x1.item<float>(), x1.grad().item<float>());
            Assert.Equal(5.0f, x2.grad().item<float>());
        }

        [Fact]
        public void TestAutoGradBackward2()
        {
            using var _ = torch.NewDisposeScope();
            var x1 = torch.rand(1, requiresGrad: true);
            var x2 = torch.rand(1, requiresGrad: true);

            var y = x1.pow(2) + 5 * x2;

            y.backward(new[] { torch.ones_like(y) });
            Assert.Equal(x1.shape, x1.grad().shape);
            Assert.Equal(x2.shape, x2.grad().shape);
            Assert.Equal(2.0f * x1.item<float>(), x1.grad().item<float>());
            Assert.Equal(5.0f, x2.grad().item<float>());
        }
    }
}
