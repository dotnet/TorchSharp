using TorchSharp;
using Xunit;

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestDisposeScopesStatisticsPackedSequenceScoped : TestDisposeScopesStatisticsBase
    {

        //ATTENTION WEIRDNESS HERE!
        //See notes in TestDisposeScopesStatisticsPackedSequenceUnscoped before modifying tests here.
        //The same weird behavior occurs in this fixture. Additionally since scopes are created
        //before the test data in this fixture, you will always see the internal tensors
        //having been detached and the data tensors created and disposed in scope.
        public TestDisposeScopesStatisticsPackedSequenceScoped()
        {
            ResetStats();
        }

        [Fact]
        public void CreatingIncrementsCreatedInScope()
        {
            using var scope = torch.NewDisposeScope();
            using var ps = TestDisposeScopesStatisticsPackedSequenceUnscoped.Create();
            AssertTensorCounts(0, 0, 6, 2, 0, 4, 4);
            AssertPackedCounts(0, 0, 1, 0, 0, 0, 1);
            AssertTotalsCounts(0, 0, 7, 2, 0, 4, 5);
        }

        [Fact]
        public void AttachingIncrementsNothing()
        {
            using var scope = torch.NewDisposeScope();
            using var ps = TestDisposeScopesStatisticsPackedSequenceUnscoped.Create();
            scope.Attach(ps);
            AssertTensorCounts(0, 0, 6, 2, 0, 4, 4);
            AssertPackedCounts(0, 0, 1, 0, 0, 0, 1);
            AssertTotalsCounts(0, 0, 7, 2, 0, 4, 5);
        }

        [Fact]
        public void DetachingIncrementsDetached()
        {
            using var scope = torch.NewDisposeScope();
            using var ps = TestDisposeScopesStatisticsPackedSequenceUnscoped.Create();
            scope.Detach(ps);
            AssertTensorCounts(0, 0, 6, 2, 0, 4, 4);
            AssertPackedCounts(0, 0, 1, 0, 0, 1, 1);
            AssertTotalsCounts(0, 0, 7, 2, 0, 5, 5);
        }

        [Fact]
        public void DisposingIncrementsDisposed()
        {
            using var scope = torch.NewDisposeScope();
            var ps = TestDisposeScopesStatisticsPackedSequenceUnscoped.Create();
            ps.Dispose();
            AssertTensorCounts(0, 0, 6, 6, 0, 4, 0);
            AssertPackedCounts(0, 0, 1, 1, 0, 0, 0);
            AssertTotalsCounts(0, 0, 7, 7, 0, 4, 0);
            ps.Dispose();
            //Ensuring the count doesn't increment again (no re-entry)
            AssertPackedCounts(0, 0, 1, 1, 0, 0, 0);
        }

        [Fact]
        public void DisposingScopeIncrementsDisposed()
        {
            var scope = torch.NewDisposeScope();
            using var ps = TestDisposeScopesStatisticsPackedSequenceUnscoped.Create();
            scope.Dispose();
            AssertTensorCounts(0, 0, 6, 6, 0, 4, 0);
            AssertPackedCounts(0, 0, 1, 1, 0, 0, 0);
            AssertTotalsCounts(0, 0, 7, 7, 0, 4, 0);
        }

        [Fact]
        public void DisposingScopeAfterDetachingLeavesSequenceAndInternalTensorsLive()
        {
            var scope = torch.NewDisposeScope();
            using var ps = TestDisposeScopesStatisticsPackedSequenceUnscoped.Create();
            scope.Detach(ps);
            scope.Dispose();
            AssertTensorCounts(0, 0, 6, 2, 0, 4, 4);
            AssertPackedCounts(0, 0, 1, 0, 0, 1, 1);
            AssertTotalsCounts(0, 0, 7, 2, 0, 5, 5);
        }
    }
}