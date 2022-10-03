// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using TorchSharp.Modules;
using static TorchSharp.torch.nn;
using Xunit;
using Google.Protobuf;
using Tensorboard;
using static TorchSharp.torch.utils.tensorboard;
using ICSharpCode.SharpZipLib;
using System.Collections.Generic;

#nullable enable

namespace TorchSharp
{
#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
    public class TestJIT
    {


        [Fact]
        public void TestLoadJIT_Func()
        {
            // One linear layer followed by ReLU.
            using var m = torch.jit.load<torch.Tensor, torch.Tensor, torch.Tensor>(@"func.script.dat");

            var sms = m.named_modules().ToArray();
            Assert.Empty(sms);

            var kids = m.named_children().ToArray();
            Assert.Empty(kids);

            var t = m.forward(torch.ones(10), torch.ones(10));

            Assert.Equal(new long[] { 10 }, t.shape);
            Assert.Equal(torch.float32, t.dtype);
            Assert.True(torch.tensor(new float[] { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 }).allclose(t));
        }

        [Fact]
        public void TestLoadJIT_1()
        {
            // One linear layer followed by ReLU.
            using var m = torch.jit.load<torch.Tensor, torch.Tensor>(@"linrelu.script.dat");
            var t = m.forward(torch.ones(10));

            Assert.Equal(new long[] { 6 }, t.shape);
            Assert.Equal(torch.float32, t.dtype);
            Assert.True(torch.tensor(new float[] { 0.313458264f, 0, 0.9996568f, 0, 0, 0 }).allclose(t));
        }

        [Fact]
        public void TestSaveJIT()
        {
            var location = "TestSaveJIT.ts";
            if (File.Exists(location)) File.Delete(location);

            try {

                // One linear layer followed by ReLU.
                using var m1 = torch.jit.load<torch.Tensor, torch.Tensor>(@"linrelu.script.dat");

                torch.jit.save(m1, location);
                using var m2 = torch.jit.load<torch.Tensor, torch.Tensor>(location);

                var t = m2.forward(torch.ones(10));

                Assert.Equal(new long[] { 6 }, t.shape);
                Assert.Equal(torch.float32, t.dtype);
                Assert.True(torch.tensor(new float[] { 0.313458264f, 0, 0.9996568f, 0, 0, 0 }).allclose(t));

            } finally {
                if (File.Exists(location)) File.Delete(location);
            }
        }

        [Fact]
        public void TestLoadJIT_2()
        {
            // One linear layer followed by ReLU.
            using var m = torch.jit.load<torch.Tensor, torch.Tensor>(@"scripted.script.dat");
            var t = m.forward(torch.ones(6));

            Assert.Equal(new long[] { 6 }, t.shape);
            Assert.Equal(torch.float32, t.dtype);
            Assert.True(torch.tensor(new float[] { 1.554085f, 1.01024628f, -1.35086036f, -1.84021854f, 0.0127189457f, 0.5994258f }).allclose(t));
        }

        [Fact]
        public void TestLoadJIT_3()
        {
            // Two linear layers, nested Sequential, ReLU in between.
            using var m = torch.jit.load<torch.Tensor, torch.Tensor>(@"l1000_100_10.script.dat");

            var sms = m.named_modules().ToArray();
            Assert.Equal(4, sms.Length);

            var kids = m.named_children().ToArray();
            Assert.Equal(2, kids.Length);

            var t = m.forward(torch.ones(1000));

            Assert.Equal(new long[] { 10 }, t.shape);
            Assert.Equal(torch.float32, t.dtype);
            Assert.True(torch.tensor(new float[] { 0.564213157f, -0.04519982f, -0.005117342f, 0.395530462f, -0.3780813f, -0.004734449f, -0.3221216f, -0.289159119f, 0.268511474f, 0.180702567f }).allclose(t));

            Assert.Throws<System.Runtime.InteropServices.ExternalException>(() => m.forward(torch.ones(100)));
        }

        [Fact]
        public void TestLoadJIT_4()
        {
            // Definitely not a TorchScript file. Let's see what the runtime does with it.
            Assert.Throws<System.Runtime.InteropServices.ExternalException>(() => torch.jit.load(@"bug510.dat"));
        }

        [Fact]
        public void TestSaveLoadJITCUDA()
        {
            if (torch.cuda.is_available()) {

                using var m = torch.jit.load<torch.Tensor, torch.Tensor>(@"linrelu.script.dat");

                m.to(DeviceType.CUDA);
                var params0 = m.parameters().ToArray();
                foreach (var p in params0)
                    Assert.Equal(DeviceType.CUDA, p.device_type);

                var t = m.forward(torch.ones(10).cuda()).cpu();

                Assert.Equal(new long[] { 6 }, t.shape);
                Assert.Equal(torch.float32, t.dtype);
                Assert.True(torch.tensor(new float[] { 0.313458264f, 0, 0.9996568f, 0, 0, 0 }).allclose(t));
            }
        }

        [Fact]
        public void TestJIT_TupleOut()
        {
            // def a(x, y):
            //     return x + y, x - y
            //
            using var m = torch.jit.load<(torch.Tensor, torch.Tensor)>(@"tuple_out.dat");

            var x = torch.rand(3, 4);
            var y = torch.rand(3, 4);
            var output = m.forward(x, y);

            Assert.Multiple(
            () => Assert.Equal(x.shape, output.Item1.shape),
            () => Assert.Equal(x.shape, output.Item2.shape),
            () => Assert.Equal(x + y, output.Item1),
            () => Assert.Equal(x - y, output.Item2)
            );
        }

        [Fact]
        public void TestJIT_TupleOutError()
        {
            // def a(x, y):
            //     return x + y, x - y
            //
            using var m = torch.jit.load< (torch.Tensor, torch.Tensor)>(@"func.script.dat");

            var x = torch.rand(3, 4);
            var y = torch.rand(3, 4);
            Assert.Throws<InvalidCastException>(() => m.forward(x, y));
        }

        [Fact]
        public void TestJIT_ListOut()
        {
            // def a(x, y):
            //     return [x + y, x - y]
            //
            using var m = torch.jit.load<torch.Tensor[]>(@"list_out.dat");

            var x = torch.rand(3, 4);
            var y = torch.rand(3, 4);
            var output = m.forward(x, y);

            Assert.Multiple(
            () => Assert.Equal(x.shape, output[0].shape),
            () => Assert.Equal(x.shape, output[1].shape),
            () => Assert.Equal(x + y, output[0]),
            () => Assert.Equal(x - y, output[1])
            );
        }

        [Fact]
        public void TestJIT_ListOutError()
        {
            // def a(x, y):
            //     return x + y, x - y
            //
            using var m = torch.jit.load<torch.Tensor[]>(@"func.script.dat");

            var x = torch.rand(3, 4);
            var y = torch.rand(3, 4);
            Assert.Throws<InvalidCastException>(() => m.forward(x, y));
        }
    }
}
