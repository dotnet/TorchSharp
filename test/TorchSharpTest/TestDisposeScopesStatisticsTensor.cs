using Xunit;

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestDisposeScopesStatisticsTensorScoped : TestDisposeScopesStatisticsBase
    {
        [Fact]
        public void CreatingIncrementsCreatedInScopeOnce()
        {
            using var scope = torch.NewDisposeScope();
            ResetStats();
            var t = torch.tensor(3);
            AssertStatCounts(1, 0, 0, 0, 1);
        }

        [Fact]
        public void AttachingIncrementsNothing()
        {
            using var scope = torch.NewDisposeScope();
            var t = torch.tensor(3);
            ResetStats();
            scope.Attach(t);
            AssertStatCounts(0, 0, 0, 0, 0);
        }

        [Fact]
        public void DetachingIncrementsDetachedOnce()
        {
            using var scope = torch.NewDisposeScope();
            var t = torch.tensor(3);
            ResetStats();
            scope.Detach(t);
            AssertStatCounts(0, 0, 1, 0, -1);
        }

        [Fact]
        public void DisposingIncrementsDisposedOnce()
        {
            using var scope = torch.NewDisposeScope();
            var t = torch.tensor(3);
            ResetStats();
            t.Dispose();
            AssertStatCounts(0, 0, 0, 1, -1);
        }

        [Fact]
        public void DisposingScopeIncrementsDisposedOnce()
        {
            var scope = torch.NewDisposeScope();
            var t = torch.tensor(3);
            ResetStats();
            scope.Dispose();
            AssertStatCounts(0, 0, 0, 1, -1);
        }

        [Fact]
        public void DisposingScopeAfterDetachingDoesNothing()
        {
            var scope = torch.NewDisposeScope();
            var t = torch.tensor(3);
            scope.Detach(t);
            ResetStats();
            scope.Dispose();
            AssertStatCounts(0, 0, 0, 0, 0);
        }
    }
}