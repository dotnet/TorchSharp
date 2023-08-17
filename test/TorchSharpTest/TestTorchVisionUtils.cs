using System.IO;
using Xunit;
using TorchSharp;

namespace TorchVision
{
    public class TestTorchVisionUtils
    {
        [Fact]
        public void Save_Image_TestMemoryUsage()
        {
            var imager = new torchvision.io.SkiaImager();
            using var ms = new MemoryStream();
            using var image = torch.randn(32, 3, 32, 32);
            using (var d = torch.NewDisposeScope()) {
                torchvision.utils.save_image(image, ms, torchvision.ImageFormat.Png, imager: imager);
                Assert.Equal(0, d.DisposablesCount);
            }
        }

        [Fact]
        public void Make_Grid_IncorrectInput()
        {
            Assert.Throws<System.Runtime.InteropServices.ExternalException>(() => {
                using var image = torch.tensor(new[] { 1.0f, 0.0f });
                using var result = torchvision.utils.make_grid(image);
            });
        }

        [Fact]
        public void Make_Grid_ImageInput()
        {
            using var image = torch.tensor(new[,] { { 1.0f, 0.0f }, { 0.0f, 1.0f } });
            using var result = torchvision.utils.make_grid(image);
            Assert.Equal(new long[] { 3, 2, 2 }, result.shape);

            using var expected = torch.tensor(new[, ,] {
                { { 1.0f, 0.0f }, { 0.0f, 1.0f } },
                { { 1.0f, 0.0f }, { 0.0f, 1.0f } },
                { { 1.0f, 0.0f }, { 0.0f, 1.0f } }
            });

            Assert.Equal(expected, result);
        }

        [Fact]
        public void Make_Grid_ColorImageInput()
        {
            using var image = torch.tensor(new[, ,] {
                { { 1.0f, 0.0f }, { 0.0f, 1.0f } },
                { { 1.0f, 0.0f }, { 0.0f, 1.0f } },
                { { 1.0f, 0.0f }, { 0.0f, 1.0f } }
            });
            using var result = torchvision.utils.make_grid(image);
            Assert.Equal(new long[] { 3, 2, 2 }, result.shape);

            Assert.Equal(image, result);
        }

        [Fact]
        public void Make_Grid_BatchColorImageInput()
        {
            using var image = torch.tensor(new[, , ,] {{
                    { { 1.0f, 0.0f }, { 0.0f, 1.0f } },
                    { { 1.0f, 0.0f }, { 0.0f, 1.0f } },
                    { { 1.0f, 0.0f }, { 0.0f, 1.0f } }
                },
                {
                    { { 1.0f, 0.0f }, { 0.0f, 1.0f } },
                    { { 1.0f, 0.0f }, { 0.0f, 1.0f } },
                    { { 1.0f, 0.0f }, { 0.0f, 1.0f } }
                }
            });
            using var result = torchvision.utils.make_grid(image, padding: 0);
            Assert.Equal(new long[] { 3, 2, 4 }, result.shape);

            using var expected = torch.tensor(new[, ,] {
                { { 1.0f, 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } },
                { { 1.0f, 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } },
                { { 1.0f, 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } },
            });

            Assert.Equal(expected, result);
        }
    }
}
