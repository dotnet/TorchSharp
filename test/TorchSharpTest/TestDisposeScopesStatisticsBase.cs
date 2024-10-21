using Xunit;

namespace TorchSharp
{
    public class TestDisposeScopesStatisticsBase
    {
        protected static void ResetStats()
        {
            DisposeScopeManager.Statistics.Reset();
            AssertStatCounts(0, 0, 0, 0, 0);
        }

        protected static void AssertStatCounts(long createdInScope, long createdOutsideScope, long detachedFrom, long disposedIn, long threadTotalLive)
        {
            var stats = DisposeScopeManager.Statistics;
            Assert.True(createdInScope == stats.CreatedInScopeCount, $"CreatedInScope: Expected({createdInScope})!={stats.CreatedInScopeCount}");
            Assert.True(createdOutsideScope == stats.CreatedOutsideScopeCount, $"CreatedOutsideScopeCount: Expected({createdOutsideScope})!={stats.CreatedOutsideScopeCount}");
            Assert.True(detachedFrom == stats.DetachedFromScopeCount, $"DetachedFromScopeCount: Expected({detachedFrom})!={stats.DetachedFromScopeCount}");
            Assert.True(disposedIn == stats.DisposedInScopeCount, $"DisposedInScopeCount: Expected({disposedIn})!={stats.DisposedInScopeCount}");
            Assert.True(threadTotalLive == stats.ThreadTotalLiveCount, $"ThreadTotalLiveCount: Expected({threadTotalLive})!={stats.ThreadTotalLiveCount}");
        }
    }
}