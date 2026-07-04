// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable

using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using static TorchSharp.torch;
using Xunit;

namespace TorchSharp
{
    /// <summary>
    /// Fact attribute that only runs when locally-generated .pt2 models are present.
    /// Run generate_export_models.py to create the models before running these tests.
    /// </summary>
    public sealed class ExportModelFactAttribute : FactAttribute
    {
        public ExportModelFactAttribute(string modelFile)
        {
            if (!File.Exists(modelFile))
            {
                Skip = $"Model file '{modelFile}' not found. Run generate_export_models.py to create test models.";
            }
        }
    }

    [Collection("Sequential")]
    public class TestExport
    {
        // ---- Platform-independent API surface tests (always run in CI) ----

        [Fact]
        public void TestExport_LoadNonExistentFile()
        {
            Assert.Throws<ExternalException>(() =>
                torch.export.load("nonexistent.pt2"));
        }

        [Fact]
        public void TestExport_LoadInvalidFile()
        {
            var tmpFile = Path.GetTempFileName();
            try
            {
                File.WriteAllBytes(tmpFile, new byte[] { 0xDE, 0xAD, 0xBE, 0xEF });
                Assert.ThrowsAny<Exception>(() =>
                    torch.export.load(tmpFile));
            }
            finally
            {
                File.Delete(tmpFile);
            }
        }

        [Fact]
        public void TestExport_LoadEmptyPath()
        {
            Assert.ThrowsAny<Exception>(() =>
                torch.export.load(""));
        }

        [Fact]
        public void TestExport_GenericLoadNonExistentFile()
        {
            Assert.Throws<ExternalException>(() =>
                torch.export.load<Tensor>("nonexistent.pt2"));
        }

        [Fact]
        public void TestExport_GenericLoadInvalidFile()
        {
            var tmpFile = Path.GetTempFileName();
            try
            {
                File.WriteAllBytes(tmpFile, new byte[] { 0xDE, 0xAD, 0xBE, 0xEF });
                Assert.ThrowsAny<Exception>(() =>
                    torch.export.load<Tensor>(tmpFile));
            }
            finally
            {
                File.Delete(tmpFile);
            }
        }

        // ---- Local-only model tests (require generate_export_models.py) ----

        [ExportModelFact("simple_linear.export.pt2")]
        public void TestLoadExport_SimpleLinear()
        {
            using var exported = torch.export.load("simple_linear.export.pt2");
            Assert.NotNull(exported);

            var input = torch.ones(10);
            var results = exported.run(input);

            Assert.NotNull(results);
            Assert.Single(results);
            Assert.Equal(new long[] { 5 }, results[0].shape);
            Assert.Equal(torch.float32, results[0].dtype);
        }

        [ExportModelFact("linrelu.export.pt2")]
        public void TestLoadExport_LinearReLU()
        {
            using var exported = torch.export.load<Tensor>("linrelu.export.pt2");
            Assert.NotNull(exported);

            var input = torch.ones(10);
            var result = exported.call(input);

            Assert.Equal(new long[] { 6 }, result.shape);
            Assert.Equal(torch.float32, result.dtype);
            Assert.True(result.data<float>().All(v => v >= 0));
        }

        [ExportModelFact("two_inputs.export.pt2")]
        public void TestLoadExport_TwoInputs()
        {
            using var exported = torch.export.load("two_inputs.export.pt2");
            Assert.NotNull(exported);

            var input1 = torch.ones(10);
            var input2 = torch.ones(10) * 2;
            var results = exported.forward(input1, input2);

            Assert.NotNull(results);
            Assert.Single(results);
            Assert.Equal(new long[] { 10 }, results[0].shape);

            var expected = torch.ones(10) * 3;
            Assert.True(expected.allclose(results[0]));
        }

        [ExportModelFact("tuple_out.export.pt2")]
        public void TestLoadExport_TupleOutput()
        {
            using var exported = torch.export.load<(Tensor, Tensor)>("tuple_out.export.pt2");
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

        [ExportModelFact("list_out.export.pt2")]
        public void TestLoadExport_ListOutput()
        {
            using var exported = torch.export.load<Tensor[]>("list_out.export.pt2");
            Assert.NotNull(exported);

            var x = torch.rand(3, 4);
            var y = torch.rand(3, 4);
            var result = exported.forward(x, y);

            Assert.IsType<Tensor[]>(result);
            Assert.Equal(2, result.Length);

            Assert.True((x + y).allclose(result[0]));
            Assert.True((x - y).allclose(result[1]));
        }

        [ExportModelFact("three_out.export.pt2")]
        public void TestLoadExport_ThreeOutputs()
        {
            using var exported = torch.export.load<(Tensor, Tensor, Tensor)>("three_out.export.pt2");
            Assert.NotNull(exported);

            var x = torch.rand(3, 4);
            var y = torch.rand(3, 4);
            var result = exported.call(x, y);

            Assert.IsType<ValueTuple<Tensor, Tensor, Tensor>>(result);
            var (sum, diff, prod) = result;

            Assert.Equal(x.shape, sum.shape);
            Assert.Equal(x.shape, diff.shape);
            Assert.Equal(x.shape, prod.shape);
            Assert.True((x + y).allclose(sum));
            Assert.True((x - y).allclose(diff));
            Assert.True((x * y).allclose(prod));
        }

        [ExportModelFact("sequential.export.pt2")]
        public void TestLoadExport_Sequential()
        {
            using var exported = torch.export.load<Tensor>("sequential.export.pt2");
            Assert.NotNull(exported);

            var input = torch.ones(1000);
            var result = exported.call(input);

            Assert.Equal(new long[] { 10 }, result.shape);
            Assert.Equal(torch.float32, result.dtype);
        }
    }
}
