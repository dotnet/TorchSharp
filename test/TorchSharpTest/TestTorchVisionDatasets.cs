using System.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision.datasets;
using Xunit;
using System.Collections.Generic;
using System;

namespace TorchSharp
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
        public void TestCeleba()
        {
            var my_transform = new MyTransform();
            using (var dataset = CelebA(".", download: true)) {
            }
            using (var dataset = CelebA(".", transform: my_transform)) {
                var loader = new DataLoader(dataset, 10, shuffle: true);
                foreach (var batch in loader) {
                    Console.WriteLine("{0}", batch);
                }
            }
        }
    }
}
