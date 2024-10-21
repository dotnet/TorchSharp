using Xunit;

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestDisposeScopesStatisticsTensorUnscoped : TestDisposeScopesStatisticsBase
    {

        [Fact]
        public void CreatingIncrementsCreatedOutsideScope()
        {
            ResetStats();
            var t = torch.tensor(3);
            AssertStatCounts(0, 1, 0, 0, 0);
        }

        [Fact]
        public void AttachingIncrementsNothing()
        {
            var t = torch.tensor(3);
            ResetStats();
            using var scope = torch.NewDisposeScope();
            scope.Attach(t);
            AssertStatCounts(0, 0, 0, 0, 0);
        }

        [Fact]
        public void DetachingIncrementsNothing()
        {
            var t = torch.tensor(3);
            ResetStats();
            using var scope = torch.NewDisposeScope();
            scope.Detach(t);
            AssertStatCounts(0, 0, 0, 0, 0);
        }

        [Fact]
        public void DisposingIncrementsNothing()
        {
            var t = torch.tensor(3);
            ResetStats();
            t.Dispose();
            AssertStatCounts(0, 0, 0, 0, 0);
        }

        [Fact]
        public void DisposingAttachedIncrementsDisposed()
        {
            var t = torch.tensor(3);
            using var scope = torch.NewDisposeScope();
            scope.Attach(t);
            ResetStats();
            t.Dispose();
            AssertStatCounts(0, 0, 0, 1, -1);
        }

        [Fact]
        public void DisposingScopeWithAttachedIncrementsDisposed()
        {
            var t = torch.tensor(3);
            var scope = torch.NewDisposeScope();
            scope.Attach(t);
            ResetStats();
            scope.Dispose();
            AssertStatCounts(0, 0, 0, 1, -1);
        }

        [Fact]
        public void DetachingAttachedIncrementsDetached()
        {
            var t = torch.tensor(3);
            using var scope = torch.NewDisposeScope();
            scope.Attach(t);
            ResetStats();
            scope.Detach(t);
            AssertStatCounts(0, 0, 1, 0, -1);
        }

        [Fact]
        public void DisposingScopeAfterDetachingDoesNothing()
        {
            var t = torch.tensor(3);
            var scope = torch.NewDisposeScope();
            scope.Attach(t);
            scope.Detach(t);
            ResetStats();
            scope.Dispose();
            AssertStatCounts(0, 0, 0, 0, 0);
        }
    }
}