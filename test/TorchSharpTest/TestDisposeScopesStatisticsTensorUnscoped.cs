using Xunit;

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestDisposeScopesStatisticsTensorUnscoped : TestDisposeScopesStatisticsBase
    {

        //When reviewing these tests, refer first to TestDisposeScopesStatisticsTensor. The tests here are
        //essentially identical, except that the numbers all look weird because a PackedSequence is constructed
        //from Tensors, and contains Tensors that it detaches from any scope and manages itself.


        [Fact]
        public void CreatingPackedSequenceIncrementsCreatedOutsideScope()
        {
            ResetStats();
            var t = torch.tensor(3);
            AssertStatCounts(0, 1, 0, 0, 0);
        }

        [Fact]
        public void AttachingPackedSequenceIncrementsNothing()
        {
            var t = torch.tensor(3);
            ResetStats();
            using var scope = torch.NewDisposeScope();
            scope.Attach(t);
            AssertStatCounts(0, 0, 0, 0, 0);
        }

        [Fact]
        public void DetachingPackedSequenceIncrementsNothing()
        {
            var t = torch.tensor(3);
            ResetStats();
            using var scope = torch.NewDisposeScope();
            scope.Detach(t);
            AssertStatCounts(0, 0, 0, 0, 0);
        }

        [Fact]
        public void DisposingPackedSequenceIncrementsNothing()
        {
            var t = torch.tensor(3);
            ResetStats();
            t.Dispose();
            AssertStatCounts(0, 0, 0, 0, 0);
        }

        [Fact]
        public void DisposingAttachedPackedSequenceIncrementsDisposed()
        {
            var t = torch.tensor(3);
            using var scope = torch.NewDisposeScope();
            scope.Attach(t);
            ResetStats();
            t.Dispose();
            AssertStatCounts(0, 0, 0, 1, -1);
        }

        [Fact]
        public void DisposingScopeWithAttachedPackedSequenceIncrementsDisposed()
        {
            var t = torch.tensor(3);
            var scope = torch.NewDisposeScope();
            scope.Attach(t);
            ResetStats();
            scope.Dispose();
            AssertStatCounts(0, 0, 0, 1, -1);
        }

        [Fact]
        public void DetachingAttachedPackedSequenceIncrementsDetached()
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