using Xunit;
using TorchSharp;

namespace TorchVision
{
    public class TestTorchVisionTransforms
    {
        private readonly torch.Tensor image = torch.tensor(new[, , ,]
        {
            {
                { { 1, 0 }, { 0, 1 } },
                { { 1, 0 }, { 0, 1 } },
                { { 1, 0 }, { 0, 1 } }
            },
            {
                { { 1, 0 }, { 0, 1 } },
                { { 1, 0 }, { 0, 1 } },
                { { 1, 0 }, { 0, 1 } }
            }
        }, dtype: torch.uint8);

        [Fact]
        public void RandAugment_TestMemoryUsage()
        {
            using (var d = torch.NewDisposeScope()) {
                var transform = torchvision.transforms.RandAugment();
                var result = transform.call(image);
                Assert.Equal(1, d.DisposablesCount);
                result?.Dispose();
                Assert.Equal(0, d.DisposablesCount);
            }
        }

        [Fact]
        public void RandAugment_TestAugment()
        {
            /* Seed 3 applies this order of transforms:
             * AutoContrast
             * Posterize
             * Identity
             * ShearX */
            var g = new torch.Generator();
            g.manual_seed(3);

            var transform = torchvision.transforms.RandAugment(generator: g);

            var result = transform.call(image);

            // Verified expected results from pytorch torchvision
            var expected = torch.tensor(new[, , ,]
            {
                {
                    { { 254, 0 }, { 0, 254 } },
                    { { 254, 0 }, { 0, 254 } },
                    { { 254, 0 }, { 0, 254 } }
                },
                {
                    { { 254, 0 }, { 0, 254 } },
                    { { 254, 0 }, { 0, 254 } },
                    { { 254, 0 }, { 0, 254 } }
                }
            }, dtype: torch.uint8);

            Assert.Equal(expected, result);
        }

        [Fact]
        public void AugMix_TestMemoryUsage()
        {
            using (var d = torch.NewDisposeScope()) {
                var transform = torchvision.transforms.AugMix();
                var result = transform.call(image);
                Assert.Equal(1, d.DisposablesCount);
                result?.Dispose();
                Assert.Equal(0, d.DisposablesCount);
            }
        }

        [Fact]
        public void AugMix_TestAugment()
        {
            var g = new torch.Generator();
            g.manual_seed(3);

            var transform = torchvision.transforms.AugMix(generator: g);

            var result = transform.call(image);

            // Verified expected results from pytorch torchvision
            var expected = torch.tensor(new[, , ,]
            {
                {
                    {{8, 0}, {0, 8}},
                    {{8, 0}, {0, 8}},
                    {{8, 0}, {0, 8}}
                },
                {
                    {{8, 0}, {0, 8}},
                    {{8, 0}, {0, 8}},
                    {{8, 0}, {0, 8}}
                }
            }, dtype: torch.uint8);

            Assert.Equal(expected, result);
        }
    }
}
