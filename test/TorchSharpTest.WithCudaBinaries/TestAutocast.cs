using System;
using TorchSharp;
using TorchSharp.Amp;
using Xunit;

using static TorchSharp.torch;
namespace TorchSharpTest.WithCudaBinaries
{
    public class TestAutocast
    {
        private static void CheckCUDA()
        {
            if (!torch.cuda_is_available())
                throw new Exception("CUDA IS NOT AVAILABLE");
        }
        [Fact]
        [TestOf("AutocastF16")]
        public void TestAutocastF16()
        {
            CheckCUDA();
            var a = torch.rand(3, 2, 4, ScalarType.Float32, new Device(DeviceType.CUDA));
            var b = torch.rand(3, 2, 4, ScalarType.Float32, new Device(DeviceType.CUDA));
            var vec1 = torch.rand(3, ScalarType.Float32, new Device(DeviceType.CUDA));
            var vec2 = torch.rand(3, ScalarType.Float32, new Device(DeviceType.CUDA));
            using (AutocastMode.GetInstance().Enter()) {
                var c = a.matmul(b);
                var d = a.addbmm(b, b);
                var e = a.baddbmm(b, b);
                var f = a.addmm(b, b);
                var g = a.addr(vec1, vec2);
                var h = a.mm(b);
                var i = a.mv(vec1);
                var j = a.bmm(b);
                Assert.Equal(ScalarType.Float16,c.dtype);
                Assert.Equal(ScalarType.Float16,d.dtype);
                Assert.Equal(ScalarType.Float16,e.dtype);
                Assert.Equal(ScalarType.Float16,f.dtype);
                Assert.Equal(ScalarType.Float16,g.dtype);
                Assert.Equal(ScalarType.Float16,h.dtype);
                Assert.Equal(ScalarType.Float16,i.dtype);
                Assert.Equal(ScalarType.Float16,j.dtype);
            }

            /*Assert.Equal(ScalarType.Float16, c.dtype);
            Assert.Equal(ScalarType.Float16, d.dtype);
            Assert.Equal(ScalarType.Float16, e.dtype);
            Assert.Equal(ScalarType.Float16, f.dtype);
            Assert.Equal(ScalarType.Float16, g.dtype);
            Assert.Equal(ScalarType.Float16, h.dtype);
            Assert.Equal(ScalarType.Float16, i.dtype);
            Assert.Equal(ScalarType.Float16, j.dtype);*/
            throw new NotImplementedException();
        }

        [Fact]
        [TestOf("AutocastF16")]
        public void TestAutocastF16Arithmetic()
        {
            //Like matmul, addmm, mm, mv, etc.
            throw new NotImplementedException();
        }

        [Fact]
        [TestOf("AutocastF16")]
        public void TestAutocastF16Cell()
        {
            //Like GRUCell, LSTM, RNN
            throw new NotImplementedException();
        }

        [Fact]
        [TestOf("AutocastF16")]
        public void TestAutocastF16Other()
        {
            //Like Linear, prelu, etc.
            throw new NotImplementedException();
        }



        [Fact]
        [TestOf("AutocastF16")]
        public void TestAutocastF16Convolutions()
        {
            //Conv 1d,2d,3d, conv_transpose 1d,2d,3d
            throw new NotImplementedException();
        }
        [Fact]
        [TestOf("AutocastF32")]
        public void TestAutocastF32()
        {
            CheckCUDA();
            throw new NotImplementedException();
        }

        [Fact]
        [TestOf("AutocastF32")]
        public void TestAutocastF32Trigonometry()
        {
            CheckCUDA();
            var a = torch.rand(3, 2, 4, ScalarType.Float32, new Device(DeviceType.CUDA));
            var b = torch.rand(3, 2, 4, ScalarType.Float32, new Device(DeviceType.CUDA));
            var vec1 = torch.rand(3, ScalarType.Float32, new Device(DeviceType.CUDA));
            var vec2 = torch.rand(3, ScalarType.Float32, new Device(DeviceType.CUDA));
            using (AutocastMode.GetInstance().Enter()) {
                const ScalarType f32 = ScalarType.Float32;
                var c = a.acos();
                var d = a.asin();
                var e = a.cosh();
                var f = a.tan();
                var g = a.sinh();
                Assert.Equal(f32, c.dtype);
                Assert.Equal(f32, d.dtype);
                Assert.Equal(f32, e.dtype);
                Assert.Equal(f32, f.dtype);
                Assert.Equal(f32, g.dtype);
            }
        }

        [Fact]
        [TestOf("AutocastF32")]
        public void TestAutocastF32Logarithmic()
        {
            CheckCUDA();
            var a = torch.rand(3, 2, 4, ScalarType.Float32, new Device(DeviceType.CUDA));
            var b = torch.rand(3, 2, 4, ScalarType.Float32, new Device(DeviceType.CUDA));
            var vec1 = torch.rand(3, ScalarType.Float32, new Device(DeviceType.CUDA));
            var vec2 = torch.rand(3, ScalarType.Float32, new Device(DeviceType.CUDA));
            using (AutocastMode.GetInstance().Enter()) {
                const ScalarType f32 = ScalarType.Float32;
                var c = a.log();
                var d = a.log10();
                var e = a.log_softmax(1);
                var f = a.log1p();
                var g = a.log2();
                Assert.Equal(f32, c.dtype);
                Assert.Equal(f32, d.dtype);
                Assert.Equal(f32, e.dtype);
                Assert.Equal(f32, f.dtype);
                Assert.Equal(f32, g.dtype);
            }
        }
        [Fact]
        [TestOf("AutocastF32")]
        public void TestAutocastF32Loss()
        {
            CheckCUDA();
            var a = torch.rand(3, 2, 4, ScalarType.Float32, new Device(DeviceType.CUDA));
            var b = torch.rand(3, 2, 4, ScalarType.Float32, new Device(DeviceType.CUDA));
            var vec1 = torch.rand(3, ScalarType.Float32, new Device(DeviceType.CUDA));
            var vec2 = torch.rand(3, ScalarType.Float32, new Device(DeviceType.CUDA));
            using (AutocastMode.GetInstance().Enter()) {
                var c = torch.nn.L1Loss().forward(a,b);
                var d = a.log10();
                var e = a.log_softmax(1);
                var f = a.log1p();
                var g = a.log2();
            }
        }

        [Fact]
        [TestOf("AutocastFWidestType")]
        public void TestAutocastFWidest()
        {
            //addcdiv,addcmul, atan2, bilinear,cross, dot,grid_sample, index_put (not implemented in TorchSharp), scatter_add, tensordot.
            throw new NotImplementedException();
        }
    }
}
