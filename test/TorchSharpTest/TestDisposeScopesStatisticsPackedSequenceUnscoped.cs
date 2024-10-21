using TorchSharp;
using Xunit;

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestDisposeScopesStatisticsPackedSequenceUnscoped : TestDisposeScopesStatisticsBase
    {

        //When reviewing these tests, refer first to TestDisposeScopesStatisticsTensor. The tests here are
        //essentially identical, except that the numbers all look weird because a PackedSequence is constructed
        //from Tensors, and contains Tensors that it detaches from any scope and manages itself.

        public static torch.nn.utils.rnn.PackedSequence Create()
        {
            var sequences = new[] { torch.tensor(new long[] { 1, 2, 3, 4 }), torch.tensor(new long[] { 5, 6 }), };
            return torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted: true);
        }

        [Fact]
        public void CreatingIncrementsCreatedOutsideScope()
        {
            ResetStats();
            var _ = Create();
            //7 = The PackedSequence, it's 4 internal tensors, and the 2 source data tensors
            AssertStatCounts(0, 7, 0, 0, 0);
        }

        [Fact]
        public void AttachingIncrementsNothing()
        {
            var ps = Create();
            ResetStats();
            using var scope = torch.NewDisposeScope();
            scope.Attach(ps);
            AssertStatCounts(0, 0, 0, 0, 0);
        }

        [Fact]
        public void DetachingIncrementsNothing()
        {
            var ps = Create();
            ResetStats();
            using var scope = torch.NewDisposeScope();
            scope.Detach(ps);
            AssertStatCounts(0, 0, 0, 0, 0);
        }

        [Fact]
        public void DisposingIncrementsNothing()
        {
            var ps = Create();
            ResetStats();
            ps.Dispose();
            AssertStatCounts(0, 0, 0, 0, 0);
        }

        [Fact]
        public void DisposingAttachedIncrementsDisposed()
        {
            var ps = Create();
            using var scope = torch.NewDisposeScope();
            scope.Attach(ps);
            ResetStats();
            ps.Dispose();
            //5 = the PackedSequence and it's 4 internal tensors
            AssertStatCounts(0, 0, 0, 5, -5);
        }

        [Fact]
        public void DisposingScopeWithAttachedIncrementsDisposed()
        {
            var ps = Create();
            var scope = torch.NewDisposeScope();
            scope.Attach(ps);
            ResetStats();
            scope.Dispose();
            AssertStatCounts(0, 0, 0, 1, -1);
        }

        [Fact]
        public void DetachingAttachedIncrementsDetached()
        {
            var ps = Create();
            using var scope = torch.NewDisposeScope();
            scope.Attach(ps);
            ResetStats();
            scope.Detach(ps);
            AssertStatCounts(0, 0, 1, 0, -1);
        }

        [Fact]
        public void DisposingScopeAfterDetachingDoesNothing()
        {
            var ps = Create();
            var scope = torch.NewDisposeScope();
            scope.Attach(ps);
            scope.Detach(ps);
            ResetStats();
            scope.Dispose();
            AssertStatCounts(0, 0, 0, 0, 0);
        }
    }
}