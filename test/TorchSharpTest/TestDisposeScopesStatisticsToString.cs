using System;
using Xunit;

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestDisposeScopesStatisticsToString
    {

        [Fact]
        public void TestTensorStatistics()
        {
            DisposeScopeManager.Statistics.Reset();
            using var scope = torch.NewDisposeScope();
            using var t = torch.tensor(3);
            Assert.Equal("ThreadTotalLiveCount: 1; " +
                         "CreatedOutsideScopeCount: 0; DisposedOutsideScopeCount: 0; " +
                         "CreatedInScopeCount: 1; DisposedInScopeCount: 0; " +
                         "AttachedToScopeCount: 0; DetachedFromScopeCount: 0"
                , DisposeScopeManager.Statistics.TensorStatistics.ToString());
        }

        [Fact]
        public void TestPackedSequenceStatistics()
        {
            DisposeScopeManager.Statistics.Reset();
            using var scope = torch.NewDisposeScope();
            using var ps = TestDisposeScopesStatisticsPackedSequenceUnscoped.Create();
            ps.DetachFromDisposeScope();
            Assert.Equal("ThreadTotalLiveCount: 1; " +
                         "CreatedOutsideScopeCount: 0; DisposedOutsideScopeCount: 0; " +
                         "CreatedInScopeCount: 1; DisposedInScopeCount: 0; " +
                         "AttachedToScopeCount: 0; DetachedFromScopeCount: 1"
                             , DisposeScopeManager.Statistics.PackedSequenceStatistics.ToString());
        }

        [Fact]
        public void TestTotalStatistics()
        {
            DisposeScopeManager.Statistics.Reset();
            using var scope = torch.NewDisposeScope();
            using var ps = TestDisposeScopesStatisticsPackedSequenceUnscoped.Create();
            Assert.Equal("ThreadTotalLiveCount: 5; " +
                         "CreatedOutsideScopeCount: 0; DisposedOutsideScopeCount: 0; " +
                         "CreatedInScopeCount: 7; DisposedInScopeCount: 2; " +
                         "AttachedToScopeCount: 0; DetachedFromScopeCount: 4"
                , DisposeScopeManager.Statistics.ToString());
        }
   }
}