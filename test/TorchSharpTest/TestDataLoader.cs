using System;
using System.Collections.Generic;
using Xunit;


namespace TorchSharp
{
    public class TestDataLoader
    {
        private class TestDataset : torch.utils.data.Dataset
        {
            public override int Count { get; } = 10;
            public override Dictionary<string, torch.Tensor> GetTensor(int index)
            {
                return new() {{"data", torch.tensor(1)}, {"label", torch.tensor(13)}, {"index", torch.tensor(index)}};
            }
        }

        [Fact]
        public void DatasetTest()
        {
            using var dataset = new TestDataset();
            var d = dataset.GetTensor(0);
            Assert.True(d.ContainsKey("data"));
            Assert.True(d.ContainsKey("index"));
            Assert.True(d.ContainsKey("label"));

            Assert.Equal(d["data"], torch.tensor(1));
            Assert.Equal(d["label"], torch.tensor(13));
            Assert.Equal(d["index"], torch.tensor(0));
        }

        [Fact]
        public void DataLoaderTest()
        {
            using var dataset = new TestDataset();
            using var dataloader = new torch.utils.data.DataLoader(dataset, 2, false, torch.CPU);
            var iterator = dataloader.GetEnumerator();
            iterator.MoveNext();
            Assert.Equal(iterator.Current["data"], torch.tensor(rawArray: new[]{1, 1}, dimensions: new[]{2L}));
            Assert.Equal(iterator.Current["label"], torch.tensor(rawArray: new[]{13, 13}, dimensions: new[]{2L}));
            Assert.Equal(iterator.Current["index"], torch.tensor(rawArray: new[]{0, 1}, dimensions: new[]{2L}));
            iterator.Dispose();
        }
    }
}