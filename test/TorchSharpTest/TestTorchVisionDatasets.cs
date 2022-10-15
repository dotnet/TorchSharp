using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision.datasets;
using Xunit;
using System.Collections.Generic;
using TorchSharp.Data;
using System;

namespace TorchSharp
{
#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
    public class TestTorchVisionDatasets
    {
        private (Tensor, Tensor) my_transform(Tensor input, string[] target)
        {
            return (input + 10, torch.zeros(target.Length));
        }

        private (Tensor, Tensor) collate(IEnumerable<(Tensor, Tensor)> batch, torch.Device device)
        {
            var input_batch = new List<Tensor>();
            var target_batch = new List<Tensor>();
            foreach (var (input, target) in batch) {
                input_batch.Add(input);
                target_batch.Add(target);
            }
            return (torch.stack(input_batch), torch.stack(target_batch));
        }

        [Fact]
        public void TestCeleba()
        {
            using (var dataset = CelebA(".")) {
            }
            using (var dataset = CelebA(".", transforms: my_transform)) {
                var loader = new DataLoader<(Tensor, Tensor), (Tensor, Tensor)>(dataset, 10, collate_fn: collate, shuffle: true);
                foreach (var batch in loader) {
                    Console.WriteLine("{0}", batch);
                }
            }
        }
    }
}
