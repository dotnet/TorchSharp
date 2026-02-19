// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Xunit;

#nullable enable

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestExport
    {
        [Fact]
        public void TestLoadExport_SimpleLinear()
        {
            // Test loading a simple linear model (inference-only)
            using var exported = torch.export.load(@"simple_linear.export.pt2");
            Assert.NotNull(exported);

            var input = torch.ones(10);
            var results = exported.run(input);

            Assert.NotNull(results);
            Assert.Single(results);
            Assert.Equal(new long[] { 5 }, results[0].shape);
            Assert.Equal(torch.float32, results[0].dtype);
        }

        [Fact]
        public void TestLoadExport_LinearReLU()
        {
            // Test loading a Linear + ReLU model with typed output
            using var exported = torch.export.load<Tensor>(@"linrelu.export.pt2");
            Assert.NotNull(exported);

            var input = torch.ones(10);
            var result = exported.call(input);

            Assert.Equal(new long[] { 6 }, result.shape);
            Assert.Equal(torch.float32, result.dtype);

            // ReLU should zero out negative values
            Assert.True(result.data<float>().All(v => v >= 0));
        }

        [Fact]
        public void TestLoadExport_TwoInputs()
        {
            // Test loading a model with two inputs
            using var exported = torch.export.load(@"two_inputs.export.pt2");
            Assert.NotNull(exported);

            var input1 = torch.ones(10);
            var input2 = torch.ones(10) * 2;
            var results = exported.forward(input1, input2);

            Assert.NotNull(results);
            Assert.Single(results);
            Assert.Equal(new long[] { 10 }, results[0].shape);

            // Should be input1 + input2 = 1 + 2 = 3
            var expected = torch.ones(10) * 3;
            Assert.True(expected.allclose(results[0]));
        }

        [Fact]
        public void TestLoadExport_TupleOutput()
        {
            // Test loading a model that returns a tuple
            using var exported = torch.export.load<(Tensor, Tensor)>(@"tuple_out.export.pt2");
            Assert.NotNull(exported);

            var x = torch.rand(3, 4);
            var y = torch.rand(3, 4);
            var result = exported.call(x, y);

            Assert.IsType<ValueTuple<Tensor, Tensor>>(result);
            var (sum, diff) = result;

            Assert.Equal(x.shape, sum.shape);
            Assert.Equal(x.shape, diff.shape);
            Assert.True((x + y).allclose(sum));
            Assert.True((x - y).allclose(diff));
        }

        [Fact]
        public void TestLoadExport_ListOutput()
        {
            // Test loading a model that returns a list
            using var exported = torch.export.load<Tensor[]>(@"list_out.export.pt2");
            Assert.NotNull(exported);

            var x = torch.rand(3, 4);
            var y = torch.rand(3, 4);
            var result = exported.forward(x, y);

            Assert.IsType<Tensor[]>(result);
            Assert.Equal(2, result.Length);

            Assert.True((x + y).allclose(result[0]));
            Assert.True((x - y).allclose(result[1]));
        }

        [Fact]
        public void TestLoadExport_Sequential()
        {
            // Test loading a sequential model
            using var exported = torch.export.load<Tensor>(@"sequential.export.pt2");
            Assert.NotNull(exported);

            var input = torch.ones(1000);
            var result = exported.call(input);

            Assert.Equal(new long[] { 10 }, result.shape);
            Assert.Equal(torch.float32, result.dtype);
        }


        [Fact]
        public void TestExport_LoadNonExistentFile()
        {
            // Test error handling for non-existent file
            Assert.Throws<System.Runtime.InteropServices.ExternalException>(() =>
                torch.export.load(@"nonexistent.pt2"));
        }
    }
}
