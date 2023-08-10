using System.IO;
using Xunit;

namespace TorchSharp
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
    }
}
