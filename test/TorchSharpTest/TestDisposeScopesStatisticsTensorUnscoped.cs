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
    }
}