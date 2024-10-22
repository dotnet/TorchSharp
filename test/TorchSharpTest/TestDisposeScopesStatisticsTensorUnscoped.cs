using System;
using System.Threading;
using Tensorboard;
using Xunit;

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestDisposeScopesStatisticsTensorUnscoped : TestDisposeScopesStatisticsBase
    {
        public TestDisposeScopesStatisticsTensorUnscoped()
        {
            ResetStats();
        }

        [Fact]
        public void CreatingIncrementsCreatedOutsideScope()
        {
            using var t = torch.tensor(3);
            AssertTensorCounts(1, 0, 0, 0, 0, 0, 1);
            AssertTotalsCounts(1, 0, 0, 0, 0, 0, 1);
        }

        [Fact]
        public void AttachingIncrementsAttached()
        {
            //What about disposing scope was moved from with Attach?
            using var t = torch.tensor(3);
            using var scope = torch.NewDisposeScope();
            scope.Attach(t);
            AssertTensorCounts(1, 0, 0, 0, 1, 0, 1);
            AssertTotalsCounts(1, 0, 0, 0, 1, 0, 1);
        }

        [Fact]
        public void DetachingIncrementsNothingBecauseObjectIsNotInAScope()
        {
            using var t = torch.tensor(3);
            using var scope = torch.NewDisposeScope();
            scope.Detach(t);
            AssertTensorCounts(1, 0, 0, 0, 0, 0, 1);
            AssertTotalsCounts(1, 0, 0, 0, 0, 0, 1);
        }

        [Fact]
        public void DisposingIncrementsDisposedOutsideScope()
        {
            var t = torch.tensor(3);
            t.Dispose();
            AssertTensorCounts(1, 1, 0, 0, 0, 0, 0);
            AssertTotalsCounts(1, 1, 0, 0, 0, 0, 0);
            t.Dispose();
            //Ensuring the count doesn't increment again (no re-entry)
            AssertTensorCounts(1, 1, 0, 0, 0, 0, 0);
        }


        [Fact]
        public void DisposingAttachedIncrementsDisposedInScope()
        {
            var t = torch.tensor(3);
            using var scope = torch.NewDisposeScope();
            scope.Attach(t);

            t.Dispose();
            AssertTensorCounts(1, 0, 0, 1, 1, 0, 0);
            AssertTotalsCounts(1, 0, 0, 1, 1, 0, 0);
        }

        [Fact]
        public void DisposingScopeWithAttachedIncrementsDisposed()
        {
            using var t = torch.tensor(3);
            var scope = torch.NewDisposeScope();
            scope.Attach(t);
            scope.Dispose();
            AssertTensorCounts(1, 0, 0, 1, 1, 0, 0);
            AssertTotalsCounts(1, 0, 0, 1, 1, 0, 0);
        }

        [Fact]
        public void DetachingAttachedIncrementsDetached()
        {
            using var t = torch.tensor(3);
            using var scope = torch.NewDisposeScope();
            scope.Attach(t);
            scope.Detach(t);
            AssertTensorCounts(1, 0, 0, 0, 1, 1, 1);
            AssertTotalsCounts(1, 0, 0, 0, 1, 1, 1);
        }

        [Fact]
        public void DisposingScopeAfterDetachingDoesNothing()
        {
            using var t = torch.tensor(3);
            var scope = torch.NewDisposeScope();
            scope.Attach(t);
            scope.Detach(t);
            scope.Dispose();
            AssertTensorCounts(1, 0, 0, 0, 1, 1, 1);
            AssertTotalsCounts(1, 0, 0, 0, 1, 1, 1);
        }

        [Fact]
        public void AttachAgainIsSameAsMoveToOtherAndStatsDoNotChange()
        {
            using var t = torch.tensor(3);
            using var scope = torch.NewDisposeScope();
            scope.Attach(t);
            var scope2 = torch.NewDisposeScope();
            AssertTensorCounts(1, 0, 0, 0, 1, 0, 1);
            AssertTotalsCounts(1, 0, 0, 0, 1, 0, 1);
            scope2.Dispose();
            AssertTensorCounts(1, 0, 0, 0, 1, 0, 1);
            AssertTotalsCounts(1, 0, 0, 0, 1, 0, 1);
        }

        [Fact]
        public void ToTensorCreatesOrphanedTensor()
        {
            //Defect: This needs fixing but is unrelated to the commit that discovered
            //it - adding better lifetime statistics. ToTensor() leaks 1 tensor
            // //every time it is called.
            var stats = DisposeScopeManager.Statistics;
            stats.Reset();
            var a1 = 1.ToTensor();
            //Should be 1 ideally. If it remains two...
            Assert.Equal(2, stats.CreatedOutsideScopeCount);
            // ... this should be 1
            Assert.Equal(0, stats.DisposedOutsideScopeCount);
            a1.Dispose();
            //... and this should also be 2
            Assert.Equal(1, stats.DisposedOutsideScopeCount);
        }
    }
}