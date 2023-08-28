using System.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision.datasets;
using Xunit;
using System.Collections.Generic;
using TorchSharp;
using System.IO;

namespace TorchVision
{
#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
    public class TestTorchVisionDatasets
    {
        class MyTransform : Module<Tensor, Tensor>
        {
            public MyTransform() : base("MyTransform") { }

            public override Tensor forward(Tensor input)
            {
                return input + 1;
            }
        }

        [Fact]
        public void TestGDriveDownload()
        {
            string file_id = "dummy_file_id";
            string filepath = Path.GetTempFileName();
            string root = Path.GetDirectoryName(filepath);
            string filename = Path.GetFileName(filepath);
            File.WriteAllText(filepath, "test");
            try {
                string md5 = "098f6bcd4621d373cade4e832627b4f6";
                // This should not download file from GDrive and exit without exception.
                torchvision.datasets.utils.download_file_from_google_drive(
                    file_id, root, filename: filename, md5: md5);
            } finally {
                File.Delete(filepath);
            }
        }

        [Fact]
        public void TestCeleba()
        {
            // There's no data so it throws an exception.
            Assert.Throws<InvalidDataException>(() => {
                using (var dataset = CelebA(".", download: false)) {
                }
            });
        }

        [Fact]
        public void TestCelebaWithTransform()
        {
            // There's no data so it throws an exception.
            Assert.Throws<InvalidDataException>(() => {
                var my_transform = new MyTransform();
                using (var dataset = CelebA(".", transform: my_transform)) {
                    var loader = new DataLoader(dataset, 10, shuffle: true);
                    foreach (var batch in loader) {
                    }
                }
            });
        }
    }
}
