// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using Xunit;
using Xunit.Abstractions;


namespace TorchSharp
{
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
            iterator.Dispose();
        }

        [Fact]
        public void DataLoaderTest2()
        {
            using var dataset = new TestDataset();
            using var dataloader = new torch.utils.data.DataLoader(dataset, 2, false, torch.CPU);
            foreach (var x in dataloader) {
                Assert.Equal(x["data"], torch.tensor(new[]{1, 1}, new[]{2L}));
            }
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
            iter.MoveNext();
            x = iter.Current;
            Assert.Equal(x["data"], torch.tensor(new[]{1, 1, 1, 1}, new[]{4L}));
            iter.MoveNext();
            x = iter.Current;
            Assert.Equal(x["data"], torch.tensor(new[]{1, 1}, new[]{2L}));
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
            iter.MoveNext();
            x = iter.Current;
            Assert.Equal(x["data"], torch.tensor(new[]{1, 1, 1, 1, 1}, new[]{5L}));
            Assert.False(iter.MoveNext());
            iter.Dispose();
        }

        [Fact]
        public void MultiThreadDataLoaderTest3()
        {
            using var dataset = new TestDataset();
            using var dataloader = new torch.utils.data.DataLoader(dataset, 5, false, torch.CPU, num_worker: 10);
            var iter = dataloader.GetEnumerator();
            iter.MoveNext();
            var x = iter.Current;
            Assert.Equal(x["data"], torch.tensor(new[]{1, 1, 1, 1, 1}, new[]{5L}));
            iter.MoveNext();
            x = iter.Current;
            Assert.Equal(x["data"], torch.tensor(new[]{1, 1, 1, 1, 1}, new[]{5L}));
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