using System;
using TorchSharp;
using TorchSharp.Amp;
using TorchSharp.Modules;
using Xunit;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpTest.WithCudaBinaries
{
    public class TestAutocast
    {
        internal const ScalarType f32 = ScalarType.Float32;
        internal const ScalarType f16 = ScalarType.Float16;
        internal static DeviceType availableDevice;
        private static void CheckCUDA()
        {
            if (!torch.cuda_is_available()) {
                availableDevice = DeviceType.CPU;
                //throw new Exception("CUDA IS NOT AVAILABLE");
            } else {
                availableDevice= DeviceType.CUDA;
            }

            AutocastMode.GetInstance(true);
            Assert.True(AutocastMode.IsAutocastEnabled());
        }
        private Tensor randnf32cuda(long dim0)
        {
            return torch.randn(dim0, f32, new Device(availableDevice));
        }

        private Tensor randnf32cuda(long dim0, long dim1)
        {
            return torch.randn(dim0, dim1, f32, new Device(availableDevice));
        }
        private Tensor randnf32cuda(long dim0, long dim1, long dim2)
        {
            return torch.randn(dim0, dim1,dim2, f32, new Device(availableDevice));
        }
        [Fact]
        [TestOf("AutocastF16")]
        public void TestAutocastF16()
        {
            CheckCUDA();
            /*var a = torch.rand(3, 2, 4, ScalarType.Float32, new Device(DeviceType.CUDA));
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
            }*/

            /*Assert.Equal(ScalarType.Float16, c.dtype);
            Assert.Equal(ScalarType.Float16, d.dtype);
            Assert.Equal(ScalarType.Float16, e.dtype);
            Assert.Equal(ScalarType.Float16, f.dtype);
            Assert.Equal(ScalarType.Float16, g.dtype);
            Assert.Equal(ScalarType.Float16, h.dtype);
            Assert.Equal(ScalarType.Float16, i.dtype); 
            Assert.Equal(ScalarType.Float16, j.dtype);*/
            //throw new NotImplementedException();
        }

        [Fact]
        [TestOf("AutocastF16")]
        public void TestAutocastF16Arithmetic()
        {
            //Like matmul, addmm, mm, mv, etc.
            CheckCUDA();
            /*var a = randnf32cuda(3, 2, 4);
            var b = randnf32cuda(3, 2, 4);*/
            var cm = randnf32cuda(3, 2);
            var dm = randnf32cuda(2, 4);

            var M= randnf32cuda(3, 5);
            //var M1= randnf32cuda(10,3, 5);
            var batch1= randnf32cuda(10,3, 4);
            var batch2= randnf32cuda(10,4, 5);
            //var batch3= randnf32cuda(10,5, 4);

            var M2 = randnf32cuda(2, 3);
            var mat1 = randnf32cuda(2, 3);
            var mat2 = randnf32cuda(3, 3);

            var M3 = randnf32cuda(4, 3);
            var vec1 = torch.rand(4, f32, new Device(availableDevice));
            var vec2 = torch.rand(3, f32, new Device(availableDevice));
            using (AutocastMode.GetInstance().Enter()) {
                var c = cm.matmul(dm);
                var d = M.addbmm(batch1, batch2);
                //var e = batch2.baddbmm(batch3, batch3);
                var f = M2.addmm(mat1, mat2);
                var g = M3.addr(vec1, vec2);
                var h = cm.mm(dm);
                var i = M2.mv(vec2);
                var j = batch1.bmm(batch2);
                Assert.Equal(f16, c.dtype);
                Assert.Equal(f16, d.dtype);
                Assert.Equal(f16, f.dtype);
                Assert.Equal(f16, h.dtype);
                //Assert.Equal(f16, e.dtype);
                Assert.Equal(f16, f.dtype);
                Assert.Equal(f16, g.dtype);
                Assert.Equal(f16, h.dtype);
                Assert.Equal(f16, i.dtype);
                Assert.Equal(f16, j.dtype);
            }
        }

        [Fact]
        [TestOf("AutocastF16")]
        public void TestAutocastF16Cell()
        {
            CheckCUDA();
            //Like GRUCell, LSTM, RNN
            var l = Linear(4, 4).to(availableDevice);
            var gru = GRUCell(4, 4).to(availableDevice);
            var lstm = LSTMCell(10, 20).to(availableDevice);
            var rnn = RNNCell(10,20).to(availableDevice);
            
            var a = torch.rand(4,4, f32, new Device(availableDevice));
            var b = torch.rand(4,4, f32, new Device(availableDevice));
            var inpRNN = torch.rand(3,10, f32, new Device(availableDevice));
            var hx = torch.rand(3,20, f32, new Device(availableDevice));
            var cx = torch.rand(3,20, f32, new Device(availableDevice));

            Assert.Equal(f32, a.dtype);
            Assert.Equal(f32, b.dtype);
            using (AutocastMode.GetInstance().Enter()) {
                a = l.forward(a);
                b = gru.forward(b);
                (torch.Tensor d, torch.Tensor f) = lstm.forward(inpRNN, new (hx,cx));
                torch.Tensor g = rnn.forward(inpRNN, hx);
                Assert.Equal(f16, a.dtype);
                Assert.Equal(f16, b.dtype);
                Assert.Equal(f16, d.dtype);
                Assert.Equal(f16, f.dtype);
                Assert.Equal(f16, g.dtype);
            }

            //Outside should have same dtype as inside
            Assert.Equal(f16, a.dtype);
            Assert.Equal(f16, b.dtype);
            //Assert.Equal(f16, e.dtype);
        }

        [Fact]
        [TestOf("AutocastF16")]
        public void TestAutocastF16Other()
        {
            //Like Linear, prelu, etc.
            CheckCUDA();
            var pr = PReLU(8).to(availableDevice);
            var a = torch.rand(8, 8, ScalarType.Float32, new Device(availableDevice));
            Assert.Equal(f32, a.dtype);
            using (AutocastMode.GetInstance().Enter()) {
                a = pr.forward(a);
                Assert.Equal(f16, a.dtype);
            }
            //Outside should have same dtype as inside
            Assert.Equal(f16, a.dtype);
        }



        [Fact]
        [TestOf("AutocastF16")]
        public void TestAutocastF16Convolutions()
        {
            CheckCUDA();
            //Conv 1d,2d,3d, conv_transpose 1d,2d,3d
            var c1 =Conv1d(4,4, 3).to(availableDevice);
            var c2 =Conv2d(4,4, 3).to(availableDevice);
            var c3 =Conv3d(4,4, 3).to(availableDevice);

            var a = torch.rand(4, 4, f32, new Device(availableDevice));
            var b = torch.rand(4, 4,3, f32, new Device(availableDevice));
            var c = torch.rand(4, 4,4,3, f32, new Device(availableDevice));
            Assert.Equal(f32, a.dtype);
            using (AutocastMode.GetInstance().Enter()) {
                a = c1.forward(a);
                b = c2.forward(b);
                c = c3.forward(c);
                Assert.Equal(f16, a.dtype);
                Assert.Equal(f16, b.dtype);
                Assert.Equal(f16, c.dtype);
            }
            //Outside should have same dtype as inside
            Assert.Equal(f16, a.dtype);
            Assert.Equal(f16, b.dtype);
            Assert.Equal(f16, c.dtype);
        }
        [Fact]
        [TestOf("AutocastF32")]
        public void TestAutocastF32()
        {
            CheckCUDA();
            //throw new NotImplementedException();
        }

        [Fact]
        [TestOf("AutocastF32")]
        public void TestAutocastF32Trigonometry()
        {
            CheckCUDA();
            //Purpose rand f16 because inside autocast with these operations should return as f32
            var a = torch.rand(3, 2, 4, f16, new Device(availableDevice));
            /*var b = torch.rand(3, 2, 4, f16, new Device(DeviceType.CUDA));
            var vec1 = torch.rand(3, f16, new Device(DeviceType.CUDA));
            var vec2 = torch.rand(3, f16, new Device(DeviceType.CUDA));*/
            using (AutocastMode.GetInstance(true).Enter()) {
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
            var a = torch.rand(3, 2, 4, f16, new Device(availableDevice));
            /*var b = torch.rand(3, 2, 4, f16, new Device(DeviceType.CUDA));
            var vec1 = torch.rand(3, f16, new Device(DeviceType.CUDA));
            var vec2 = torch.rand(3, f16, new Device(DeviceType.CUDA));*/
            using (AutocastMode.GetInstance().Enter()) {
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
        public void TestAutocastF32Other()
        {
            CheckCUDA();
            var a = torch.rand(3, 3, f16, new Device(DeviceType.CUDA));
            //var b = torch.rand(3, 3, f32, new Device(DeviceType.CUDA));
            using (AutocastMode.GetInstance().Enter()) {
                var c = a.cumprod(1);
                Assert.Equal(f32, c.dtype);
            }
        }
        [Fact]
        [TestOf("AutocastF32")]
        public void TestAutocastF32Loss()
        {
            CheckCUDA();
            var a = torch.rand(3, 2, 4, f16, new Device(availableDevice));
            var b = torch.rand(3, 2, 4, f16, new Device(availableDevice));
            var vec1 = torch.rand(3, f16, new Device(availableDevice));
            var vec2 = torch.rand(3, f16, new Device(availableDevice));
            using (AutocastMode.AutoCastEnter()) {
                var c = torch.nn.L1Loss().to(availableDevice).forward(a,b);
                Assert.Equal(f32, c.dtype);
            }
        }

        [Fact]
        [TestOf("AutocastFWidestType")]
        public void TestAutocastFWidest()
        {
            //addcdiv,addcmul, atan2, bilinear,cross, dot,grid_sample, index_put (not implemented in TorchSharp), scatter_add, tensordot.
            //throw new NotImplementedException();
        }
    }
}
