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
#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
    public class TestJIT
    {
        [Fact]
        public void TestLoadJIT_Func()
        {
            // One linear layer followed by ReLU.
            using var m = torch.jit.load<Tensor, Tensor, Tensor>(@"func.script.dat");

            var sms = m.named_modules().ToArray();
            Assert.Empty(sms);

            var kids = m.named_children().ToArray();
            Assert.Empty(kids);

            var t = m.call(torch.ones(10), torch.ones(10));

            Assert.Equal(new long[] { 10 }, t.shape);
            Assert.Equal(torch.float32, t.dtype);
            Assert.True(torch.tensor(new float[] { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 }).allclose(t));
        }

        [Fact]
        public void TestLoadJIT_1()
        {
            // One linear layer followed by ReLU.
            using var m = torch.jit.load<Tensor, Tensor>(@"linrelu.script.dat");
            var t = m.call(torch.ones(10));

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
                using var m1 = torch.jit.load<Tensor, Tensor>(@"linrelu.script.dat");

                torch.jit.save(m1, location);
                using var m2 = torch.jit.load<Tensor, Tensor>(location);

                var t = m2.call(torch.ones(10));

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
            using var m = torch.jit.load<Tensor, Tensor>(@"scripted.script.dat");
            var t = m.call(torch.ones(6));

            Assert.Equal(new long[] { 6 }, t.shape);
            Assert.Equal(torch.float32, t.dtype);
            Assert.True(torch.tensor(new float[] { 1.554085f, 1.01024628f, -1.35086036f, -1.84021854f, 0.0127189457f, 0.5994258f }).allclose(t));
        }

        [Fact]
        public void TestLoadJIT_3()
        {
            // Two linear layers, nested Sequential, ReLU in between.
            using var m = torch.jit.load<Tensor, Tensor>(@"l1000_100_10.script.dat");

            var sms = m.named_modules().ToArray();
            Assert.Equal(4, sms.Length);

            var kids = m.named_children().ToArray();
            Assert.Equal(2, kids.Length);

            var t = m.call(torch.ones(1000));

            Assert.Equal(new long[] { 10 }, t.shape);
            Assert.Equal(torch.float32, t.dtype);
            Assert.True(torch.tensor(new float[] { 0.564213157f, -0.04519982f, -0.005117342f, 0.395530462f, -0.3780813f, -0.004734449f, -0.3221216f, -0.289159119f, 0.268511474f, 0.180702567f }).allclose(t));

            Assert.Throws<System.Runtime.InteropServices.ExternalException>(() => m.call(torch.ones(100)));
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

                {
                    using var m = torch.jit.load<Tensor, Tensor>(@"linrelu.script.dat");

                    m.to(DeviceType.CUDA);
                    var params0 = m.parameters().ToArray();
                    foreach (var p in params0)
                        Assert.Equal(DeviceType.CUDA, p.device_type);

                    var t = m.call(torch.ones(10).cuda()).cpu();

                    Assert.Equal(new long[] { 6 }, t.shape);
                    Assert.Equal(torch.float32, t.dtype);
                    Assert.True(torch.tensor(new float[] { 0.313458264f, 0, 0.9996568f, 0, 0, 0 }).allclose(t));
                }
                {
                    using var m = torch.jit.load<Tensor, Tensor>(@"linrelu.script.dat", DeviceType.CUDA);

                    var params0 = m.parameters().ToArray();
                    foreach (var p in params0)
                        Assert.Equal(DeviceType.CUDA, p.device_type);

                    var t = m.call(torch.ones(10).cuda()).cpu();

                    Assert.Equal(new long[] { 6 }, t.shape);
                    Assert.Equal(torch.float32, t.dtype);
                    Assert.True(torch.tensor(new float[] { 0.313458264f, 0, 0.9996568f, 0, 0, 0 }).allclose(t));
                }
            }
        }

        [Fact]
        public void TestJIT_TupleOut()
        {
            // def a(x, y):
            //     return x + y, x - y
            //
            using var m = torch.jit.load<(Tensor, Tensor)>(@"tuple_out.dat");

            var x = torch.rand(3, 4);
            var y = torch.rand(3, 4);
            var output = m.call(x, y);

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
            using var m = torch.jit.load<(Tensor, Tensor)>(@"func.script.dat");

            var x = torch.rand(3, 4);
            var y = torch.rand(3, 4);
            Assert.Throws<InvalidCastException>(() => m.call(x, y));
        }

        [Fact]
        public void TestJIT_ListOut()
        {
            // def a(x, y):
            //     return [x + y, x - y]
            //
            using var m = torch.jit.load<Tensor[]>(@"list_out.dat");

            var x = torch.rand(3, 4);
            var y = torch.rand(3, 4);
            var output = m.call(x, y);

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
            using var m = torch.jit.load<Tensor[]>(@"func.script.dat");

            var x = torch.rand(3, 4);
            var y = torch.rand(3, 4);
            Assert.Throws<InvalidCastException>(() => m.call(x, y));
        }



        [Fact]
        public void TestLoadJIT_Methods()
        {
            // class MyModule(nn.Module):
            //   def __init__(self):
            //     super().__init__()
            //     self.p = nn.Parameter(torch.rand(10))
            //   def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
            //     return x + y, x - y
            //
            //   @torch.jit.export
            //   def predict(self, x: Tensor) -> Tensor:
            //     return x + self.p
            //  @torch.jit.export
            //  def add_scalar(self, x: Tensor, i: int) -> Tensor:
            //     return x + i

            using var m = new TestScriptModule(@"exported.method.dat");

            var x = torch.rand(3, 4);
            var y = torch.rand(3, 4);
            var output = m.call(x, y);

            Assert.Multiple(
            () => Assert.Equal(x.shape, output.Item1.shape),
            () => Assert.Equal(x.shape, output.Item2.shape),
            () => Assert.Equal(x + y, output.Item1),
            () => Assert.Equal(x - y, output.Item2)
            );

            var ones = m.add_scalar(torch.zeros(10), 1);

            Assert.Equal(torch.ones(10), ones);

            var a = torch.rand(10);
            var predict = m.predict(a);

            Assert.Multiple(
                () => Assert.NotEqual(a, predict)
            );
        }

        internal class TestScriptModule : Module<Tensor, Tensor, (Tensor, Tensor)>
        {
            internal TestScriptModule(string filename) : base(nameof(TestScriptModule))
            {
                m = torch.jit.load<(Tensor, Tensor)> (filename);
            }

            public override (Tensor, Tensor) forward(Tensor input1, Tensor input2)
            {
                return m.call(input1, input2);
            }

            public Tensor predict(Tensor input)
            {
                return m.invoke<Tensor>("predict", input);
            }

            public Tensor add_scalar(Tensor input, int i)
            {
                return m.invoke<Tensor>("add_scalar", input, i);
            }

            private torch.jit.ScriptModule<(Tensor, Tensor)> m;
        }

        [Fact]
        public void TestJITCompile_1()
        {
            string script = @"
  def relu_script(a, b):
    return torch.relu(a + b)
  def relu6_script(a, b):
    return torch.relu6(a + b)
  def add_i(x: Tensor, i: int) -> Tensor:
    return x + i
  def add_d(x: Tensor, i: float) -> Tensor:
    return x + i
  def add_ii(x: int, i: int) -> Tuple[int,int]:
    return (x + i,x-i)
";

            using var cu = torch.jit.compile(script);

            Assert.NotNull(cu);

            var x = torch.randn(3, 4);
            var y = torch.randn(3, 4);

            var zeros = torch.zeros(3, 4);
            var ones = torch.ones(3, 4);

            var z = (Tensor)cu.invoke("relu_script", x, y);
            Assert.Equal(torch.nn.functional.relu(x + y), z);
            z = cu.invoke<Tensor>("relu6_script", x, y);
            Assert.Equal(torch.nn.functional.relu6(x + y), z);
            z = cu.invoke<Tensor>("add_i", zeros, 1);
            Assert.Equal(ones, z);
            z = cu.invoke<Tensor>("add_d", zeros, 1.0);
            Assert.Equal(ones, z);

            var ss = cu.invoke<(Scalar,Scalar)>("add_ii", 3, 1);
            Assert.Multiple(
                () => Assert.Equal(4, ss.Item1.ToInt32()),
                () => Assert.Equal(2, ss.Item2.ToInt32())
            );
        }



        [Fact]
        public void TestJITCompile_2()
        {
            string script = @"
  def none_script(a: Any, b: Any):
    return a
  def none_tuple(a: Any, b: Any):
    return (a, None)
";

            using var cu = torch.jit.compile(script);

            Assert.NotNull(cu);

            var x = torch.randn(3, 4);
            var y = torch.randn(3, 4);

            var z = cu.invoke("none_script", null, null);
            Assert.Null(z);
            z = cu.invoke("none_script", null, y);
            Assert.Null(z);
            z = cu.invoke("none_script", x, null);
            Assert.NotNull(z);

            var zArr = cu.invoke<object[]>("none_tuple", null, null);
            Assert.NotNull(zArr);
            Assert.Null(zArr[0]);
            Assert.Null(zArr[1]);

            zArr = cu.invoke<object[]>("none_tuple", x, null);
            Assert.NotNull(z);
            Assert.NotNull(zArr[0]);
            Assert.Null(zArr[1]);
        }
    }
}
