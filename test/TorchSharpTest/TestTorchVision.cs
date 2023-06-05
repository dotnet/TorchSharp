using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Xunit;

using static TorchSharp.torch;
using static TorchSharp.torchvision.models;
using static TorchSharp.torchvision.ops;

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestTorchVision
    {

        [Fact]
        public void TestBoxConvert()
        {
            var input = tensor(new float[] { 0, 0, 100, 100, 0, 0, 0, 0, 10, 15, 30, 35, 23, 35, 93, 95 }).reshape(4, 4);
            {
                // Test no-op conversion.
                var expected = tensor(new float[] { 0, 0, 100, 100, 0, 0, 0, 0, 10, 15, 30, 35, 23, 35, 93, 95 }).reshape(4, 4);
                Assert.Equal(expected, box_convert(input, BoxFormats.xyxy, BoxFormats.xyxy));
                Assert.Equal(expected, box_convert(input, BoxFormats.xywh, BoxFormats.xywh));
                Assert.Equal(expected, box_convert(input, BoxFormats.cxcywh, BoxFormats.cxcywh));
            }
            {
                // Test xyxy -> xywh and reverse.
                var expected = tensor(new float[] { 0, 0, 100, 100, 0, 0, 0, 0, 10, 15, 20, 20, 23, 35, 70, 60 }).reshape(4, 4);
                var output = box_convert(input, BoxFormats.xyxy, BoxFormats.xywh);
                Assert.Equal(expected, output);

                var back_again = box_convert(output, BoxFormats.xywh, BoxFormats.xyxy);
                Assert.Equal(input, back_again);
            }
            {
                // Test xyxy -> cxcywh and reverse.
                var expected = tensor(new float[] { 50, 50, 100, 100, 0, 0, 0, 0, 20, 25, 20, 20, 58, 65, 70, 60 }).reshape(4, 4);
                var output = box_convert(input, BoxFormats.xyxy, BoxFormats.cxcywh);
                Assert.Equal(expected, output);

                Assert.Equal(input, box_convert(output, BoxFormats.cxcywh, BoxFormats.xyxy));
            }
            {
                // Test xywh -> cxcywh and reverse.
                input = tensor(new float[] { 0, 0, 100, 100, 0, 0, 0, 0, 10, 15, 20, 20, 23, 35, 70, 60 }).reshape(4, 4);
                var expected = tensor(new float[] { 50, 50, 100, 100, 0, 0, 0, 0, 20, 25, 20, 20, 58, 65, 70, 60 }).reshape(4, 4);
                var output = box_convert(input, BoxFormats.xywh, BoxFormats.cxcywh);
                Assert.Equal(expected, output);

                Assert.Equal(input, box_convert(output, BoxFormats.cxcywh, BoxFormats.xywh));
            }
        }

        [Fact]
        public void TestBoxArea()
        {
            {
                var box_tensor = tensor(new int[] { 0, 0, 100, 100, 0, 0, 0, 0 }, dtype: int16).reshape(2, 4);
                var expected = new int[] { 10000, 0 };

                var output = box_area(box_tensor);
                Assert.Equal(expected, output.data<int>().ToArray());
            }
            {
                var box_tensor = tensor(new int[] { 0, 0, 100, 100, 0, 0, 0, 0 }, dtype: int32).reshape(2, 4);
                var expected = new int[] { 10000, 0 };

                var output = box_area(box_tensor);
                Assert.Equal(expected, output.data<int>().ToArray());
            }
            {
                var box_tensor = tensor(new int[] { 0, 0, 100, 100, 0, 0, 0, 0 }, dtype: int64).reshape(2, 4);
                var expected = new long[] { 10000, 0 };

                var output = box_area(box_tensor);
                Assert.Equal(expected, output.data<long>().ToArray());
            }
            {
                var box_tensor = tensor(new double[] { 285.3538, 185.5758, 1193.5110, 851.4551, 285.1472, 188.7374, 1192.4984, 851.0669, 279.2440, 197.9812, 1189.4746, 849.2019 }, dtype: float32).reshape(3, 4);
                var expected = tensor(new float[] { 604723.0806f, 600965.4666f, 592761.0085f });

                var output = box_area(box_tensor);
                Assert.True(expected.allclose(output));
            }
            {
                var box_tensor = tensor(new double[] { 285.3538, 185.5758, 1193.5110, 851.4551, 285.1472, 188.7374, 1192.4984, 851.0669, 279.2440, 197.9812, 1189.4746, 849.2019 }, dtype: float64).reshape(3, 4);
                var expected = tensor(new double[] { 604723.0806, 600965.4666, 592761.0085 });

                var output = box_area(box_tensor);
                Assert.True(expected.allclose(output));
            }
        }

        private (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) get_boxes(ScalarType dtype, Device device)
        {
            var box1 = tensor(new int[] { -1, -1, 1, 1 }, dtype: dtype, device: device);
            var box2 = tensor(new int[] { 0, 0, 1, 1 }, dtype: dtype, device: device);
            var box3 = tensor(new int[] { 0, 1, 1, 2 }, dtype: dtype, device: device);
            var box4 = tensor(new int[] { 1, 1, 2, 2 }, dtype: dtype, device: device);

            var box1s = stack(new[] { box2, box2 }, dim: 0);
            var box2s = stack(new[] { box3, box4 }, dim: 0);

            return (box1, box2, box3, box4, box1s, box2s);
        }

        // Using a delegate type instead of Func<...> allows us to rely on default arguments.

        private delegate Tensor LossFunc(Tensor boxes1, Tensor boxes2, nn.Reduction reduction = nn.Reduction.None, double eps = 1e-7);

        private void assert_iou_loss(LossFunc iou_fn, Tensor box1, Tensor box2, double expected_loss, Device device, nn.Reduction reduction = nn.Reduction.None)
        {
            var loss = iou_fn(box1, box2, reduction);
            var expected = tensor(expected_loss, dtype: loss.dtype, device: device);
            expected.allclose(loss);
        }

        [Fact]
        public void TestGeneralizedBoxIouLoss()
        {
            var (box1, box2, box3, box4, box1s, box2s) = get_boxes(float32, CPU);
            assert_iou_loss(generalized_box_iou_loss, box1, box1, 0.0, CPU, nn.Reduction.None);
            assert_iou_loss(generalized_box_iou_loss, box1, box2, 0.75, CPU, nn.Reduction.None);
            assert_iou_loss(generalized_box_iou_loss, box2, box3, 1.0, CPU, nn.Reduction.None);
            assert_iou_loss(generalized_box_iou_loss, box2, box4, 1.5, CPU, nn.Reduction.None);

            assert_iou_loss(generalized_box_iou_loss, box1s, box2s, 2.5, CPU, nn.Reduction.Sum);
            assert_iou_loss(generalized_box_iou_loss, box1s, box2s, 1.25, CPU, nn.Reduction.Mean);
        }

        [Fact]
        public void TestCompleteBoxIouLoss()
        {
            var (box1, box2, box3, box4, box1s, box2s) = get_boxes(float32, CPU);
            assert_iou_loss(complete_box_iou_loss, box1, box1, 0.0, CPU, nn.Reduction.None);
            assert_iou_loss(complete_box_iou_loss, box1, box2, 0.8125, CPU, nn.Reduction.None);
            assert_iou_loss(complete_box_iou_loss, box1, box3, 1.1923, CPU, nn.Reduction.None);
            assert_iou_loss(complete_box_iou_loss, box1, box4, 1.2500, CPU, nn.Reduction.None);

            assert_iou_loss(complete_box_iou_loss, box1s, box2s, 1.2250, CPU, nn.Reduction.Sum);
            assert_iou_loss(complete_box_iou_loss, box1s, box2s, 2.4500, CPU, nn.Reduction.Mean);
        }

        [Fact]
        public void TestDistanceBoxIouLoss()
        {
            var (box1, box2, box3, box4, box1s, box2s) = get_boxes(float32, CPU);
            assert_iou_loss(distance_box_iou_loss, box1, box1, 0.0, CPU, nn.Reduction.None);
            assert_iou_loss(distance_box_iou_loss, box1, box2, 0.8125, CPU, nn.Reduction.None);
            assert_iou_loss(distance_box_iou_loss, box1, box3, 1.1923, CPU, nn.Reduction.None);
            assert_iou_loss(distance_box_iou_loss, box1, box4, 1.2500, CPU, nn.Reduction.None);

            assert_iou_loss(distance_box_iou_loss, box1s, box2s, 1.2250, CPU, nn.Reduction.Sum);
            assert_iou_loss(distance_box_iou_loss, box1s, box2s, 2.4500, CPU, nn.Reduction.Mean);
        }

        private void RunBoxIoUTest(Func<Tensor, Tensor, Tensor> target_fn, Tensor actual_box1, Tensor actual_box2, Tensor expected)
        {
            var output = target_fn(actual_box1, actual_box2);
            expected.allclose(output);
        }

        private static readonly Tensor INT_BOXES = torch.tensor(new int[] { 0, 0, 100, 100, 0, 0, 50, 50, 200, 200, 300, 300, 0, 0, 25, 25 }).reshape(4, 4);
        private static readonly Tensor INT_BOXES2 = torch.tensor(new int[] { 0, 0, 100, 100, 0, 0, 50, 50, 200, 200, 300, 300 }).reshape(3, 4);
        private static readonly Tensor FLOAT_BOXES = torch.tensor(new float[] {
                285.3538f, 185.5758f, 1193.5110f, 851.4551f,
                285.1472f, 188.7374f, 1192.4984f, 851.0669f,
                279.2440f, 197.9812f, 1189.4746f, 849.2019f
            }).reshape(3, 4);

        [Fact]
        public void TestBoxIou()
        {
            foreach (var device in TestUtils.AvailableDevices()) {
                var int_expected = torch.tensor(new float[] { 1.0f, 0.25f, 0.0f, 0.25f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0625f, 0.25f, 0.0f }, device: device).reshape(4, 3);
                var flt_expected = torch.tensor(new float[] { 1.0f, 0.9933f, 0.9673f, 0.9933f, 1.0f, 0.9737f, 0.9673f, 0.9737f, 1.0f }, device: device).reshape(3, 3);

                RunBoxIoUTest(box_iou, INT_BOXES.to(device), INT_BOXES2.to(device), int_expected);
                RunBoxIoUTest(box_iou, INT_BOXES.to(device).@long(), INT_BOXES2.to(device).@long(), int_expected);
                RunBoxIoUTest(box_iou, FLOAT_BOXES.to(device), FLOAT_BOXES.to(device), flt_expected);
                RunBoxIoUTest(box_iou, FLOAT_BOXES.to(device).@double(), FLOAT_BOXES.to(device).@double(), flt_expected.to(device).@double());
            }
        }

        [Fact]
        public void TestGeneralizedBoxIou()
        {
            foreach (var device in TestUtils.AvailableDevices()) {
                var int_expected = torch.tensor(new[] { 1.0f, 0.25f, -0.7778f, 0.25f, 1.0f, -0.8611f, -0.7778f, -0.8611f, 1.0f, 0.0625f, 0.25f, -0.8819f }, device: device, dtype: float32).reshape(4, 3);
                var flt_expected = torch.tensor(new[] { 1.0, 0.9933, 0.9673, 0.9933, 1.0, 0.9737, 0.9673, 0.9737, 1.0 }, device: device, dtype: float32).reshape(3, 3);

                RunBoxIoUTest(generalized_box_iou, INT_BOXES.to(device), INT_BOXES2.to(device), int_expected);
                RunBoxIoUTest(generalized_box_iou, INT_BOXES.to(device).@long(), INT_BOXES2.to(device).@long(), int_expected);
                RunBoxIoUTest(generalized_box_iou, FLOAT_BOXES.to(device), FLOAT_BOXES.to(device), flt_expected);
                RunBoxIoUTest(generalized_box_iou, FLOAT_BOXES.to(device).@double(), FLOAT_BOXES.to(device).@double(), flt_expected.@double());
            }
        }

        [Fact]
        public void TestDistanceBoxIoU()
        {
            foreach (var device in TestUtils.AvailableDevices()) {
                var int_expected = torch.tensor(new[] { 1.0000, 0.1875, -0.4444, 0.1875, 1.0000, -0.5625, -0.4444, -0.5625, 1.0000, -0.0781, 0.1875, -0.6267 }, device: device, dtype: float32).reshape(4, 3);
                var flt_expected = torch.tensor(new[] { 1.0, 0.9933, 0.9673, 0.9933, 1.0, 0.9737, 0.9673, 0.9737, 1.0 }, device: device, dtype: float32).reshape(3, 3);

                RunBoxIoUTest((a, b) => distance_box_iou(a, b), INT_BOXES.to(device), INT_BOXES2.to(device), int_expected);
                RunBoxIoUTest((a, b) => distance_box_iou(a, b), INT_BOXES.to(device).@long(), INT_BOXES2.to(device).@long(), int_expected);
                RunBoxIoUTest((a, b) => distance_box_iou(a, b), FLOAT_BOXES.to(device), FLOAT_BOXES.to(device), flt_expected);
                RunBoxIoUTest((a, b) => distance_box_iou(a, b), FLOAT_BOXES.to(device).@double(), FLOAT_BOXES.to(device).@double(), flt_expected.@double());
            }
        }

        [Fact]
        public void TestCompleteBoxIou()
        {
            foreach (var device in TestUtils.AvailableDevices()) {
                var int_expected = torch.tensor(new[] { 1.0f, 0.25f, -0.7778f, 0.25f, 1.0f, -0.8611f, -0.7778f, -0.8611f, 1.0f, 0.0625f, 0.25f, -0.8819f }, device: device, dtype: float32).reshape(4, 3);
                var flt_expected = torch.tensor(new[] { 1.0, 0.9933, 0.9673, 0.9933, 1.0, 0.9737, 0.9673, 0.9737, 1.0 }, device: device, dtype: float32).reshape(3, 3);

                RunBoxIoUTest((a, b) => complete_box_iou(a, b), INT_BOXES.to(device), INT_BOXES2.to(device), int_expected);
                RunBoxIoUTest((a, b) => complete_box_iou(a, b), INT_BOXES.to(device).@long(), INT_BOXES2.to(device).@long(), int_expected);
                RunBoxIoUTest((a, b) => complete_box_iou(a, b), FLOAT_BOXES.to(device), FLOAT_BOXES.to(device), flt_expected);
                RunBoxIoUTest((a, b) => complete_box_iou(a, b), FLOAT_BOXES.to(device).@double(), FLOAT_BOXES.to(device).@double(), flt_expected.@double());
            }
        }

        [Fact]
        public void TestMasksToBoxes()
        {
            using var _ = torch.NewDisposeScope();
            var maskList = new[] {  0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0,
                                    0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0,
                                    0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0,
                                    0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0};
            var expected = new[] { 1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 2.0, 2.0,
                                   1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 2.0, 2.0 };
            var expected_shape = new long[] { 12, 4 };

            var types = new[] { float32, float64 };

            foreach (var device in TestUtils.AvailableDevices()) {
                foreach (var dtype in types) {
                    var output = masks_to_boxes(torch.tensor(maskList, dtype: dtype, device: device).reshape(12,3,4));
                    var exp = torch.tensor(expected, dtype: dtype, device: device).reshape(12,4);
                    Assert.Equal(expected_shape, output.shape);
                    Assert.Equal(exp, output);
                }
            }
        }

        private void TestDropBlocks(int dim, double p, int block_size, bool inplace)
        {
            int batch_size = 5;
            int channels = 3;
            long height = 11;
            long width = height;
            long depth = height;

            Tensor x = (dim == 2) ? torch.ones(batch_size, channels, height, width) : torch.ones(new[] { batch_size, channels, depth, height, width });
            nn.Module<Tensor,Tensor> layer = (dim == 2) ?  DropBlock2d(p, block_size, inplace) : DropBlock3d(p, block_size, inplace);

            int feature_size = (int)((dim == 2) ? height * width : depth * height * width);

            var output = layer.call(x);

            if (p == 0) {
                Assert.Equal(x, output);
            }
            if (block_size == height) {
                foreach (var b in Enumerable.Range(0, batch_size)) {
                    foreach (var c in Enumerable.Range(0, channels)) {
                        var nz = output[b, c].count_nonzero().item<long>();
                        Assert.InRange(nz, 0, feature_size);
                    }

                }
            }
        }

        [Fact]
        public void TestDropBlock()
        {
            foreach (var dim in new int[] { 2, 3 }) {
                foreach (var p in new double[] { 0, 0.5}) {
                    foreach (var block_size in new int[] { 5, 11}) {
                        TestDropBlocks(dim, p, block_size, false);
                        TestDropBlocks(dim, p, block_size, true);
                    }
                }
            }
        }

        [Fact]
        public void TestStochasticDepth()
        {
            using var input = torch.ones(4, 250, 250);
            var size = input.NumberOfElements;

            {
                // With p == 0, nothing should happen
                using var output = stochastic_depth(input, 0, torchvision.StochasticDepth.Mode.Batch, true);
                Assert.Equal(size, output.count_nonzero().item<long>());
            }
            {
                // With training == false, nothing should happen
                using var output = stochastic_depth(input, 1, torchvision.StochasticDepth.Mode.Batch, false);
                Assert.Equal(size, output.count_nonzero().item<long>());
            }
            {
                // If training and p == 1, then all elements should be cleared.
                using var output = stochastic_depth(input, 1, torchvision.StochasticDepth.Mode.Batch, true);
                Assert.Equal(0, output.count_nonzero().item<long>());
            }
            {
                // If training and p in ]0,1[, either all or none of the elements should be cleared.
                using var output = stochastic_depth(input, 0.5, torchvision.StochasticDepth.Mode.Batch, true);
                var nz = output.count_nonzero().item<long>();
                Assert.True(nz == 0 || nz == size);
            }
        }

        [Fact]
        public void TestFrozenBatchNorm2d()
        {
            foreach (var device in TestUtils.AvailableDevices()) {
                {
                    using var ones = torch.ones(new long[] { 16, 3, 28, 28 }, device: device);
                    using (var pool = FrozenBatchNorm2d(3, device: device)) {
                        var pooled = pool.call(ones);
                        Assert.Equal(ones.shape, pooled.shape);
                        Assert.Throws<ArgumentException>(() => pool.call(torch.ones(new long[] { 16, 2, 2 }, device: device)));
                        Assert.Throws<ArgumentException>(() => pool.call(torch.ones(new long[] { 2, 2, 2, 2, 2 }, device: device)));
                    }
                }
                {
                    using var ones = torch.ones(new long[] { 1, 3, 28, 28 }, device: device);
                    using (var pool = FrozenBatchNorm2d(3, device: device)) {
                        var pooled = pool.call(ones);
                        Assert.Equal(ones.shape, pooled.shape);
                    }
                }
            }
        }

        [Fact]
        public void TestSqueezeExcitation()
        {
            foreach (var device in TestUtils.AvailableDevices()) {
                {
                    using var _ = NewDisposeScope();
                    var ones = torch.ones(new long[] { 16, 3, 28, 28 }, device: device);
                    using (var pool = SqueezeExcitation(3, 4).to(device)) {
                        var pooled = pool.call(ones);
                        Assert.Equal(ones.shape, pooled.shape);
                        ones = torch.ones(new long[] { 3, 28, 28 }, device: device);
                        pooled = pool.call(ones);
                        Assert.Equal(ones.shape, pooled.shape);

                        Assert.Throws<ArgumentException>(() => pool.call(torch.ones(new long[] { 16, 2, 2 }, device: device)));
                        Assert.Throws<ArgumentException>(() => pool.call(torch.ones(new long[] { 16, 4, 28, 28 }, device: device)));
                        Assert.Throws<ArgumentException>(() => pool.call(torch.ones(new long[] { 2, 2, 2, 2, 2 }, device: device)));
                    }
                }
            }
        }

        [Fact]
        public void TestResNet18()
        {
            using var model = resnet18();
            var sd = model.state_dict();
            Assert.Equal(122, sd.Count);

            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("bn1", names[1]),
                () => Assert.Equal("relu", names[2]),
                () => Assert.Equal("maxpool", names[3]),
                () => Assert.Equal("layer1", names[4]),
                () => Assert.Equal("layer2", names[5]),
                () => Assert.Equal("layer3", names[6]),
                () => Assert.Equal("layer4", names[7]),
                () => Assert.Equal("avgpool", names[8]),
                () => Assert.Equal("flatten", names[9]),
                () => Assert.Equal("fc", names[10])
            );

            using var input = torch.randn(8, 3, 416, 416);
            var output = model.call(input);

            Assert.Equal(new long[] { 8, 1000 }, output.shape);
        }

        [Fact]
        public void TestResNet34()
        {
            using var model = resnet34();
            var sd = model.state_dict();            
            Assert.Equal(218, sd.Count);

            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("bn1", names[1]),
                () => Assert.Equal("relu", names[2]),
                () => Assert.Equal("maxpool", names[3]),
                () => Assert.Equal("layer1", names[4]),
                () => Assert.Equal("layer2", names[5]),
                () => Assert.Equal("layer3", names[6]),
                () => Assert.Equal("layer4", names[7]),
                () => Assert.Equal("avgpool", names[8]),
                () => Assert.Equal("flatten", names[9]),
                () => Assert.Equal("fc", names[10])
            );

            using var input = torch.randn(8, 3, 416, 416);
            using var output = model.call(input);

            Assert.Equal(new long[] { 8, 1000 }, output.shape);
        }

        [Fact(Skip = "The test takes too long to run and causes trouble in CI/CD, since it uses a lot of memory.")]
        public void TestResNet50()
        {
            using var input = torch.randn(8, 3, 416, 416);
            {
                using var model = resnet50();
                var sd = model.state_dict();
                Assert.Equal(320, sd.Count);

                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("conv1", names[0]),
                    () => Assert.Equal("bn1", names[1]),
                    () => Assert.Equal("relu", names[2]),
                    () => Assert.Equal("maxpool", names[3]),
                    () => Assert.Equal("layer1", names[4]),
                    () => Assert.Equal("layer2", names[5]),
                    () => Assert.Equal("layer3", names[6]),
                    () => Assert.Equal("layer4", names[7]),
                    () => Assert.Equal("avgpool", names[8]),
                    () => Assert.Equal("flatten", names[9]),
                    () => Assert.Equal("fc", names[10])
                );

                using var output = model.call(input);
                Assert.Equal(new long[] { 8, 1000 }, output.shape);
            }
#if false // Requires a lot of physical memory to run.
            {
                using var model = resnext50_32x4d();
                using var output = model.call(input);
                Assert.Equal(new long[] { 8, 1000 }, output.shape);
            }
            {
                using var model = wide_resnet50_2();
                var output = model.call(input);
                Assert.Equal(new long[] { 8, 1000 }, output.shape);
            }
#endif
        }

        [Fact(Skip = "The test takes too long to run and causes trouble in CI/CD, since it uses a lot of memory.")]
        public void TestResNet101()
        {
            using var input = torch.randn(8, 3, 416, 416);
            {
                using var model = resnet101();
                var sd = model.state_dict();
                Assert.Equal(626, sd.Count);

                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("conv1", names[0]),
                    () => Assert.Equal("bn1", names[1]),
                    () => Assert.Equal("relu", names[2]),
                    () => Assert.Equal("maxpool", names[3]),
                    () => Assert.Equal("layer1", names[4]),
                    () => Assert.Equal("layer2", names[5]),
                    () => Assert.Equal("layer3", names[6]),
                    () => Assert.Equal("layer4", names[7]),
                    () => Assert.Equal("avgpool", names[8]),
                    () => Assert.Equal("flatten", names[9]),
                    () => Assert.Equal("fc", names[10])
                );

                using var output = model.call(input);
                Assert.Equal(new long[] { 8, 1000 }, output.shape);
            }
        }

        [Fact(Skip = "The test takes too long to run and causes trouble in CI/CD, since it uses a lot of memory.")]
        public void TestResNet101Alt()
        {
            using var input = torch.randn(8, 3, 416, 416);
            {
                using var model = resnext101_32x8d();
                using var output = model.call(input);
                Assert.Equal(new long[] { 8, 1000 }, output.shape);
            }
            {
                using var model = resnext101_64x4d();
                using var output = model.call(input);
                Assert.Equal(new long[] { 8, 1000 }, output.shape);
            }
#if false // Requires more than 16GB of physical memory to run.
            {
                using var model = wide_resnet101_2();
                var output = model.call(input);
                Assert.Equal(new long[] { 8, 1000 }, output.shape);
            }
#endif
        }

        [Fact(Skip = "The test takes too long to run and causes trouble in CI/CD, since it uses a lot of memory.")]
        public void TestResNet152()
        {
            using var model = resnet152();
            var sd = model.state_dict();
            Assert.Equal(932, sd.Count);

            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("bn1", names[1]),
                () => Assert.Equal("relu", names[2]),
                () => Assert.Equal("maxpool", names[3]),
                () => Assert.Equal("layer1", names[4]),
                () => Assert.Equal("layer2", names[5]),
                () => Assert.Equal("layer3", names[6]),
                () => Assert.Equal("layer4", names[7]),
                () => Assert.Equal("avgpool", names[8]),
                () => Assert.Equal("flatten", names[9]),
                () => Assert.Equal("fc", names[10])
            );

            using var input = torch.randn(8, 3, 416, 416);
            using var output = model.call(input);

            Assert.Equal(new long[] { 8, 1000 }, output.shape);
        }

        [Fact]
        public void TestAlexNet()
        {
            using var model = alexnet();
            var sd = model.state_dict();
            Assert.Equal(16, sd.Count);
            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("features", names[0]),
                () => Assert.Equal("avgpool", names[1]),
                () => Assert.Equal("classifier", names[2])
            );

            using var input = torch.randn(8, 3, 416, 416);
            using var output = model.call(input);

            Assert.Equal(new long[] { 8, 1000 }, output.shape);
        }

        [Fact(Skip = "The test takes too long to run -- across the various VGG versions, 2/3 of overall test time is spent here.")]
        public void TestVGG11()
        {
            {
                using var model = vgg11();
                var sd = model.state_dict();
                Assert.Equal(22, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
            {
                using var model = vgg11_bn();
                var sd = model.state_dict();
                Assert.Equal(62, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
        }

        [Fact(Skip = "The test takes too long to run -- across the various VGG versions, 2/3 of overall test time is spent here.")]
        public void TestVGG13()
        {
            {
                using var model = vgg13();
                var sd = model.state_dict();
                Assert.Equal(26, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
            {
                using var model = vgg13_bn();
                var sd = model.state_dict();
                Assert.Equal(76, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
        }

        [Fact(Skip = "The test takes too long to run -- across the various VGG versions, 2/3 of overall test time is spent here.")]
        public void TestVGG16()
        {
            {
                using var model = vgg16();
                var sd = model.state_dict();
                Assert.Equal(32, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
            {
                using var model = vgg16_bn();
                var sd = model.state_dict();
                Assert.Equal(97, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
        }

        [Fact(Skip = "The test takes too long to run -- across the various VGG versions, 2/3 of overall test time is spent here.")]
        public void TestVGG19()
        {
            {
                using var model = vgg19();
                var sd = model.state_dict();
                Assert.Equal(38, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
            {
                using var model = vgg19_bn();
                var sd = model.state_dict();
                Assert.Equal(118, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
        }

        [Fact]
        public void TestInception()
        {
            using var model = inception_v3();
            var sd = model.state_dict();
            Assert.Equal(580, sd.Count);
            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("Conv2d_1a_3x3", names[0]),
                () => Assert.Equal("Conv2d_2a_3x3", names[1]),
                () => Assert.Equal("Conv2d_2b_3x3", names[2]),
                () => Assert.Equal("maxpool1", names[3]),
                () => Assert.Equal("Conv2d_3b_1x1", names[4]),
                () => Assert.Equal("Conv2d_4a_3x3", names[5]),
                () => Assert.Equal("maxpool2", names[6]),
                () => Assert.Equal("Mixed_5b", names[7]),
                () => Assert.Equal("Mixed_5c", names[8]),
                () => Assert.Equal("Mixed_5d", names[9]),
                () => Assert.Equal("Mixed_6a", names[10]),
                () => Assert.Equal("Mixed_6b", names[11]),
                () => Assert.Equal("Mixed_6c", names[12]),
                () => Assert.Equal("Mixed_6d", names[13]),
                () => Assert.Equal("Mixed_6e", names[14]),
                () => Assert.Equal("AuxLogits", names[15]),
                () => Assert.Equal("Mixed_7a", names[16]),
                () => Assert.Equal("Mixed_7b", names[17]),
                () => Assert.Equal("Mixed_7c", names[18]),
                () => Assert.Equal("avgpool", names[19]),
                () => Assert.Equal("dropout", names[20]),
                () => Assert.Equal("fc", names[21])
            );

            using var input = torch.randn(8, 3, 416, 416);
            using var output = model.call(input);

            Assert.Equal(new long[] { 8, 1000 }, output.shape);
        }

        [Fact]
        public void TestGoogLeNet()
        {
            using var model = googlenet();
            var sd = model.state_dict();
            Assert.Equal(344, sd.Count);
            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("maxpool1", names[1]),
                () => Assert.Equal("conv2", names[2]),
                () => Assert.Equal("conv3", names[3]),
                () => Assert.Equal("maxpool2", names[4]),
                () => Assert.Equal("inception3a", names[5]),
                () => Assert.Equal("inception3b", names[6]),
                () => Assert.Equal("maxpool3", names[7]),
                () => Assert.Equal("inception4a", names[8]),
                () => Assert.Equal("inception4b", names[9]),
                () => Assert.Equal("inception4c", names[10]),
                () => Assert.Equal("inception4d", names[11]),
                () => Assert.Equal("inception4e", names[12]),
                () => Assert.Equal("maxpool4", names[13]),
                () => Assert.Equal("inception5a", names[14]),
                () => Assert.Equal("inception5b", names[15]),
                () => Assert.Equal("avgpool", names[16]),
                () => Assert.Equal("dropout", names[17]),
                () => Assert.Equal("fc", names[18])
            );

            using var input = torch.randn(8, 3, 416, 416);
            using var output = model.call(input);

            Assert.Equal(new long[] { 8, 1000 }, output.shape);
        }

        [Fact]
        public void TestMobileNetV2()
        {
            using var model = mobilenet_v2();
            var sd = model.state_dict();
            Assert.Equal(314, sd.Count);
            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("classifier", names[0]),
                () => Assert.Equal("features", names[1])
            );

            using var input = torch.randn(8, 3, 416, 416);
            using var output = model.call(input);

            Assert.Equal(new long[] { 8, 1000 }, output.shape);
        }

        [Fact]
        public void TestMobileNetV3()
        {
            using (var model = mobilenet_v3_large()) {
                var sd = model.state_dict();
                Assert.Equal(312, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("avgpool", names[0]),
                    () => Assert.Equal("classifier", names[1]),
                    () => Assert.Equal("features", names[2])
                );

                using var input = torch.randn(8, 3, 416, 416);
                using var output = model.call(input);

                Assert.Equal(new long[] { 8, 1000 }, output.shape);
            }

            using (var model = mobilenet_v3_small()) {
                var sd = model.state_dict();
                Assert.Equal(244, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("avgpool", names[0]),
                    () => Assert.Equal("classifier", names[1]),
                    () => Assert.Equal("features", names[2])
                );

                using var input = torch.randn(8, 3, 416, 416);
                using var output = model.call(input);

                Assert.Equal(new long[] { 8, 1000 }, output.shape);
            }
        }

        [Fact]
        public void TestReadingAndWritingImages()
        {
            var fileName = "vslogo.jpg";
            var outName1 = $"TestReadingAndWritingImages_1_{fileName}";
            var outName2 = $"TestReadingAndWritingImages_2_{fileName}";

            if (System.IO.File.Exists(outName1)) System.IO.File.Delete(outName1);
            if (System.IO.File.Exists(outName2)) System.IO.File.Delete(outName2);

            torchvision.io.DefaultImager = new torchvision.io.SkiaImager(100);

            using var img = torchvision.io.read_image(fileName);
            Assert.NotNull(img);
            Assert.Equal(uint8, img.dtype);
            //Assert.Equal(new long[] { 3, 508, 728 }, img.shape);

            torchvision.io.write_image(img, outName1, torchvision.ImageFormat.Jpeg);
            Assert.True(System.IO.File.Exists(outName1));

            using var img2 = torchvision.io.read_image(outName1);
            Assert.NotNull(img2);
            Assert.Equal(uint8, img2.dtype);
            Assert.Equal(img.shape, img2.shape);

            using var grey = torchvision.transforms.functional.rgb_to_grayscale(img);
            Assert.Equal(float32, grey.dtype);

            torchvision.io.write_jpeg(torchvision.transforms.functional.convert_image_dtype(grey, ScalarType.Byte), outName2);
            Assert.True(System.IO.File.Exists(outName2));

            System.IO.File.Delete(outName1);
            System.IO.File.Delete(outName2);
        }
    }
}
