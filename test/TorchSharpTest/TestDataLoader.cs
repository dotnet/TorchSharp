// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using Xunit;


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

        [Fact]
        public void DataLoaderTest1()
        {
            using var dataset = new TestDataset();
            using var dataloader = new torch.utils.data.DataLoader(dataset, 2, false, torch.CPU);
            var iterator = dataloader.GetEnumerator();
            Assert.True(iterator.MoveNext());
            Assert.Equal(iterator.Current["data"], torch.tensor(rawArray: new[]{1L, 1L}, dimensions: new[]{2L}, dtype: torch.ScalarType.Int32));
            Assert.Equal(iterator.Current["label"], torch.tensor(rawArray: new[]{13L, 13L}, dimensions: new[]{2L}, dtype: torch.ScalarType.Int32));
            Assert.Equal(iterator.Current["index"].ToString(true), torch.tensor(rawArray: new[]{0L, 1L}, dimensions: new[]{2L}, dtype: torch.ScalarType.Int64).ToString(true));
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
        }
    }
}