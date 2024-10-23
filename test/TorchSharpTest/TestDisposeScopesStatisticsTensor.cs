using Xunit;

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestDisposeScopesStatisticsTensorScoped : TestDisposeScopesStatisticsBase
    {
        public TestDisposeScopesStatisticsTensorScoped()
        {
            ResetStats();
        }

        [Fact]
        public void CreatingIncrementsCreatedInScopeOnce()
        {
            using var scope = torch.NewDisposeScope();
            using var t = torch.tensor(3);
            AssertTensorCounts(0, 0, 1, 0, 0, 0, 1);
            AssertTotalsCounts(0, 0, 1, 0, 0, 0, 1);
        }

        [Fact]
        public void AttachingToScopeAlreadyAttachedToIncrementsNothing()
        {
            using var scope = torch.NewDisposeScope();
            using var t = torch.tensor(3);
            scope.Attach(t);
            AssertTensorCounts(0, 0, 1, 0, 0, 0, 1);
            AssertTotalsCounts(0, 0, 1, 0, 0, 0, 1);
        }

        [Fact]
        public void DetachingIncrementsDetachedOnce()
        {
            using var scope = torch.NewDisposeScope();
            using var t = torch.tensor(3);
            scope.Detach(t);
            AssertTensorCounts(0, 0, 1, 0, 0, 1, 1);
            AssertTotalsCounts(0, 0, 1, 0, 0, 1, 1);
        }

        [Fact]
        public void DisposingIncrementsDisposedOnce()
        {
            using var scope = torch.NewDisposeScope();
            var t = torch.tensor(3);
            t.Dispose();
            AssertTensorCounts(0, 0, 1, 1, 0, 0, 0);
            AssertTotalsCounts(0, 0, 1, 1, 0, 0, 0);

            t.Dispose();
            //Ensuring the count doesn't increment again (no re-entry)
            AssertTensorCounts(0, 0, 1, 1, 0, 0, 0);
        }

        [Fact]
        public void DisposingScopeIncrementsDisposedOnce()
        {
            var scope = torch.NewDisposeScope();
            using var t = torch.tensor(3);
            scope.Dispose();
            AssertTensorCounts(0, 0, 1, 1, 0, 0, 0);
            AssertTotalsCounts(0, 0, 1, 1, 0, 0, 0);
        }

        [Fact]
        public void DisposingScopeAfterDetachingDoesNothing()
        {
            var scope = torch.NewDisposeScope();
            using var t = torch.tensor(3);
            scope.Detach(t);
            scope.Dispose();
            AssertTensorCounts(0, 0, 1, 0, 0, 1, 1);
            AssertTotalsCounts(0, 0, 1, 0, 0, 1, 1);
        }

        [Fact]
        public void ToTensorCreatesOrphanedTensorButDisposeScopeCleansItUp()
        {
            //Defect: This needs fixing but is unrelated to the commit that discovered
            //it - adding better lifetime statistics. ToTensor() leaks 1 tensor
            //every time it is called.
            var scope = torch.NewDisposeScope();
            var stats = DisposeScopeManager.Statistics;
            stats.Reset();
            var a1 = 1.ToTensor();
            Assert.Equal(2, stats.CreatedInScopeCount);
            //Should be 1, or can remain 0 if CreatedInScope becomes 1.
            Assert.Equal(0, stats.DisposedInScopeCount);
            a1.Dispose();
            Assert.Equal(1, stats.DisposedInScopeCount);

            //Should not need this if no orphan.
            scope.Dispose();
            Assert.Equal(2, stats.DisposedInScopeCount);
        }
    }
}