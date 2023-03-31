// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System.Collections.Generic;
using Xunit;


namespace TorchSharp
{
#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
    public class TestDataLoader
    {
        private class TestDataset : torch.utils.data.Dataset
        {
            public override long Count { get; } = 10;
            public override Dictionary<string, torch.Tensor> GetTensor(long index)
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
            Assert.Equal(d["index"], torch.tensor(0L));
        }

        [Fact]
        public void TensorDatasetTest()
        {
            var x = torch.randn(4, 12);
            var y = torch.randn(4, 16);
            using var dataset = torch.utils.data.TensorDataset(x, y);
            Assert.Equal(2, dataset.Count);

            var d = dataset.GetTensor(0);

            Assert.Equal(2, d.Count);
            Assert.Equal(x[0], d[0]);
            Assert.Equal(y[0], d[1]);
        }

        // Cannot assert index because ConcurrentBag append tensors randomly
        [Fact]
        public void DataLoaderTest1()
        {
            using var dataset = new TestDataset();
            using var dataloader = new torch.utils.data.DataLoader(dataset, 2, false, torch.CPU);
            var iterator = dataloader.GetEnumerator();
            Assert.True(iterator.MoveNext());
            Assert.Equal(iterator.Current["data"], torch.tensor(rawArray: new[]{1L, 1L}, dimensions: new[]{2L}, dtype: torch.ScalarType.Int32));
            Assert.Equal(iterator.Current["label"], torch.tensor(rawArray: new[]{13L, 13L}, dimensions: new[]{2L}, dtype: torch.ScalarType.Int32));
            Assert.Equal(iterator.Current["index"].ToString(TensorStringStyle.Julia), torch.tensor(rawArray: new[]{0L, 1L}, dimensions: new[]{2L}, dtype: torch.ScalarType.Int64).ToString(TensorStringStyle.Julia));
            iterator.Dispose();
        }

        [Fact]
        public void DataLoaderTest2()
        {
            using var dataset = new TestDataset();
            using var dataloader = new torch.utils.data.DataLoader(dataset, 2, false, torch.CPU);
            long idx = 0;
                foreach (var x in dataloader) {
                Assert.Equal(x["data"], torch.tensor(new[]{1, 1}, new[]{2L}));
                Assert.Equal(x["index"], torch.tensor(new[]{idx++, idx++}, new[]{2L}));
            }
        }

        [Fact]
        public void DataLoaderTest3()
        {
            using var dataset = new TestDataset();
            {
                using var dataloader = new torch.utils.data.DataLoader(dataset, 3, false, torch.CPU);
                Assert.Equal(4, dataloader.Count);
            }
            {
                using var dataloader = new torch.utils.data.DataLoader(dataset, 3, false, torch.CPU, drop_last: true);
                Assert.Equal(3, dataloader.Count);
            }
            {
                using var dataloader = new torch.utils.data.DataLoader(dataset, 2, false, torch.CPU, drop_last: true);
                Assert.Equal(5, dataloader.Count);
            }
        }

        private const int stressBatchSize = 32;

        private class LargeTestDataset : torch.utils.data.Dataset
        {
            public override long Count { get; } = 2*stressBatchSize;
            public override Dictionary<string, torch.Tensor> GetTensor(long index)
            {
                return new() { { "data", torch.rand(3, 512, 512) }, { "label", torch.tensor(16) }, { "index", torch.tensor(index) } };
            }
        }

        [Fact]
        public void BigDataLoaderTest3()
        {
            using var dataset = new LargeTestDataset();
            using var dataloader = new torch.utils.data.DataLoader(dataset, stressBatchSize, false, torch.CPU);
            var iter = dataloader.GetEnumerator();
            iter.MoveNext();
            var x = iter.Current;
            Assert.Equal(new long[] { stressBatchSize, 3, 512, 512 }, x["data"].shape);
            iter.MoveNext();
            x = iter.Current;
            Assert.Equal(new long[] { stressBatchSize, 3, 512, 512 }, x["data"].shape);
            Assert.False(iter.MoveNext());
            iter.Dispose();
        }

        [Fact]
        public void MultiThreadDataLoaderTest1()
        {
            using var dataset = new TestDataset();
            using var dataloader = new torch.utils.data.DataLoader(dataset, 4, false, torch.CPU, num_worker: 2);
            var iter = dataloader.GetEnumerator();
            iter.MoveNext();
            var x = iter.Current;
            Assert.Equal(x["data"], torch.tensor(new[]{1, 1, 1, 1}, new[]{4L}));
            Assert.Equal(x["index"], torch.tensor(new[]{0L, 1, 2, 3}, new[]{4L}));
            iter.MoveNext();
            x = iter.Current;
            Assert.Equal(x["data"], torch.tensor(new[]{1, 1, 1, 1}, new[]{4L}));
            Assert.Equal(x["index"], torch.tensor(new[]{4L, 5, 6, 7}, new[]{4L}));
            iter.MoveNext();
            x = iter.Current;
            Assert.Equal(x["data"], torch.tensor(new[]{1, 1}, new[]{2L}));
            Assert.Equal(x["index"], torch.tensor(new[]{8L, 9}, new[]{2L}));
            iter.Dispose();
        }

        [Fact]
        public void MultiThreadDataLoaderTest2()
        {
            using var dataset = new TestDataset();
            using var dataloader = new torch.utils.data.DataLoader(dataset, 5, false, torch.CPU, num_worker: 2);
            var iter = dataloader.GetEnumerator();
            iter.MoveNext();
            var x = iter.Current;
            Assert.Equal(x["data"], torch.tensor(new[]{1, 1, 1, 1, 1}, new[]{5L}));
            Assert.Equal(x["index"], torch.tensor(new[]{0L, 1, 2, 3, 4}, new[]{5L}));
            iter.MoveNext();
            x = iter.Current;
            Assert.Equal(x["data"], torch.tensor(new[]{1, 1, 1, 1, 1}, new[]{5L}));
            Assert.Equal(x["index"], torch.tensor(new[]{5L, 6, 7, 8, 9}, new[]{5L}));
            Assert.False(iter.MoveNext());
            iter.Dispose();
        }

        [Fact]
        public void MultiThreadDataLoaderTest3()
        {
            using var dataset = new TestDataset();
            using var dataloader = new torch.utils.data.DataLoader(dataset, 5, false, torch.CPU, num_worker: 11);
            var iter = dataloader.GetEnumerator();
            iter.MoveNext();
            var x = iter.Current;
            Assert.Equal(x["data"], torch.tensor(new[]{1, 1, 1, 1, 1}, new[]{5L}));
            Assert.Equal(x["index"], torch.tensor(new[]{0L, 1, 2, 3, 4}, new[]{5L}));
            iter.MoveNext();
            x = iter.Current;
            Assert.Equal(x["data"], torch.tensor(new[]{1, 1, 1, 1, 1}, new[]{5L}));
            Assert.Equal(x["index"], torch.tensor(new[]{5L, 6, 7, 8, 9}, new[]{5L}));
            Assert.False(iter.MoveNext());
            iter.Dispose();
        }

        [Fact]
        public void CustomSeedTest()
        {
            using var dataset = new TestDataset();
            using var dataloader = new torch.utils.data.DataLoader(dataset, 2, true, seed: 1);
            using var dataloader2 = new torch.utils.data.DataLoader(dataset, 2, true, seed: 1);
            var iterator = dataloader.GetEnumerator();
            var iterator2 = dataloader2.GetEnumerator();
            iterator.MoveNext();
            iterator2.MoveNext();
            Assert.Equal(iterator.Current, iterator2.Current);
            iterator.Dispose();
            iterator2.Dispose();
        }
    }
}