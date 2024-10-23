using Xunit;

namespace TorchSharp
{
    public class TestDisposeScopesStatisticsBase
    {
        protected static void ResetStats()
        {
            DisposeScopeManager.Statistics.Reset();
            AssertPackedCounts(0, 0, 0, 0, 0, 0, 0);
            AssertTensorCounts(0, 0, 0, 0, 0, 0, 0);
            AssertTotalsCounts(0, 0, 0, 0, 0, 0, 0);
        }

        protected static void AssertTensorCounts(long createdOutside, long disposedOutside,
            long createdIn, long disposedIn,
            long attached, long detached,
            long threadTotalLive)
        {
            AssertStatCounts(createdOutside, disposedOutside,
                createdIn, disposedIn,
                attached, detached, threadTotalLive, DisposeScopeManager.Statistics.TensorStatistics);
        }

        protected static void AssertPackedCounts(long createdOutside, long disposedOutside,
            long createdIn, long disposedIn,
            long attached, long detached,
            long threadTotalLive)
        {
            AssertStatCounts(createdOutside, disposedOutside,
                createdIn, disposedIn,
                attached, detached, threadTotalLive, DisposeScopeManager.Statistics.PackedSequenceStatistics);
        }

        protected static void AssertTotalsCounts(long createdOutside,
            long disposedOutside,
            long createdIn, long disposedIn,
            long attached,
            long detached,
            long threadTotalLive)
        {
            var stats = DisposeScopeManager.Statistics;
            Assert.True(createdOutside == stats.CreatedOutsideScopeCount, $"CreatedOutsideScopeCount: Expected({createdOutside})!={stats.CreatedOutsideScopeCount}");
            Assert.True(disposedOutside == stats.DisposedOutsideScopeCount, $"DisposedOutsideScopeCount: Expected({disposedOutside})!={stats.DisposedOutsideScopeCount}");
            Assert.True(createdIn == stats.CreatedInScopeCount, $"CreatedInScope: Expected({createdIn})!={stats.CreatedInScopeCount}");
            Assert.True(disposedIn == stats.DisposedInScopeCount, $"DisposedInScopeCount: Expected({disposedIn})!={stats.DisposedInScopeCount}");
            Assert.True(attached == stats.AttachedToScopeCount, $"AttachedToScopeCount: Expected({attached})!={stats.AttachedToScopeCount}");
            Assert.True(detached == stats.DetachedFromScopeCount, $"DetachedFromScopeCount: Expected({detached})!={stats.DetachedFromScopeCount}");
            Assert.True(threadTotalLive == stats.ThreadTotalLiveCount, $"ThreadTotalLiveCount: Expected({threadTotalLive})!={stats.ThreadTotalLiveCount}");
        }

        protected static void AssertStatCounts(long createdOutside, long disposedOutside,
            long createdIn, long disposedIn,
            long attached, long detached,
            long threadTotalLive, LifetimeStatistics stats)
        {
            Assert.True(createdOutside == stats.CreatedOutsideScopeCount, $"CreatedOutsideScopeCount: Expected({createdOutside})!={stats.CreatedOutsideScopeCount}");
            Assert.True(disposedOutside == stats.DisposedOutsideScopeCount, $"DisposedOutsideScopeCount: Expected({disposedOutside})!={stats.DisposedOutsideScopeCount}");
            Assert.True(createdIn == stats.CreatedInScopeCount, $"CreatedInScope: Expected({createdIn})!={stats.CreatedInScopeCount}");
            Assert.True(disposedIn == stats.DisposedInScopeCount, $"DisposedInScopeCount: Expected({disposedIn})!={stats.DisposedInScopeCount}");
            Assert.True(attached == stats.AttachedToScopeCount, $"AttachedToScopeCount: Expected({attached})!={stats.AttachedToScopeCount}");
            Assert.True(detached == stats.DetachedFromScopeCount, $"DetachedFromScopeCount: Expected({detached})!={stats.DetachedFromScopeCount}");
            Assert.True(threadTotalLive == stats.ThreadTotalLiveCount, $"ThreadTotalLiveCount: Expected({threadTotalLive})!={stats.ThreadTotalLiveCount}");
        }
    }
}