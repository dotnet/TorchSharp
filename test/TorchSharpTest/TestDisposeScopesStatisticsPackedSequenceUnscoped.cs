using TorchSharp;
using Xunit;

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestDisposeScopesStatisticsPackedSequenceUnscoped : TestDisposeScopesStatisticsBase
    {

        //ATTENTION WEIRDNESS HERE!
        //When reviewing these tests, refer first to TestDisposeScopesStatisticsTensor. The tests here are
        //essentially identical, except that the numbers all look weird because a PackedSequence is constructed
        //from Tensors, and contains Tensors that it detaches from any scope and manages itself. Therefore you
        //will always see 2 disposed tensors, because the Create method here disposes them after creating the
        //PackedSequence. Additionally, since the Tensors are managed internally to the PackedSequence, the numbers
        //are slightly different for some operations (for example attached a PackedSequence will not attach the
        //internal Tensors.

        public static torch.nn.utils.rnn.PackedSequence Create()
        {
            var sequences = new[] { torch.tensor(new long[] { 1, 2, 3, 4 }), torch.tensor(new long[] { 5, 6 }), };
            var packedSequence = torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted: false);
            foreach (var t in sequences) {
                t.Dispose();
            }
            return packedSequence;
        }

        public TestDisposeScopesStatisticsPackedSequenceUnscoped()
        {
            ResetStats();
        }

        [Fact]
        public void CreatingIncrementsCreatedOutsideScope()
        {
            using var _ = Create();
            AssertTensorCounts(6, 2, 0, 0, 0, 0, 4);
            AssertPackedCounts(1, 0, 0, 0, 0, 0, 1);
            AssertTotalsCounts(7, 2, 0, 0, 0, 0, 5);
        }

        [Fact]
        public void AttachingIncrementsSequenceAttachedButNotTensors()
        {
            using var ps = Create();
            using var scope = torch.NewDisposeScope();
            scope.Attach(ps);
            AssertTensorCounts(6, 2, 0, 0, 0, 0, 4);
            AssertPackedCounts(1, 0, 0, 0, 1, 0, 1);
            AssertTotalsCounts(7, 2, 0, 0, 1, 0, 5);
        }

        [Fact]
        public void DetachingIncrementsNothingBecauseObjectIsNotInAScope()
        {
            using var ps = Create();
            using var scope = torch.NewDisposeScope();
            scope.Detach(ps);
            AssertTensorCounts(6, 2, 0, 0, 0, 0, 4);
            AssertPackedCounts(1, 0, 0, 0, 0, 0, 1);
            AssertTotalsCounts(7, 2, 0, 0, 0, 0, 5);
        }

        [Fact]
        public void DisposingIncrementsDisposedOutsideScope()
        {
            var ps = Create();
            ps.Dispose();
            AssertTensorCounts(6, 6, 0, 0, 0, 0, 0);
            AssertPackedCounts(1, 1, 0, 0, 0, 0, 0);
            AssertTotalsCounts(7, 7, 0, 0, 0, 0, 0);

            ps.Dispose();
            //Ensuring the count doesn't increment again (no re-entry)
            AssertPackedCounts(1, 1, 0, 0, 0, 0, 0);
        }

        [Fact]
        public void DisposingAttachedIncrementsDisposed()
        {
            var ps = Create();
            using var scope = torch.NewDisposeScope();
            scope.Attach(ps);
            ps.Dispose();
            AssertTensorCounts(6, 2, 0, 4, 0, 0, 0);
            AssertPackedCounts(1, 0, 0, 1, 1, 0, 0);
            AssertTotalsCounts(7, 2, 0, 5, 1, 0, 0);
        }

        [Fact]
        public void DisposingScopeWithAttachedIncrementsDisposed()
        {
            using var ps = Create();
            var scope = torch.NewDisposeScope();
            scope.Attach(ps);
            scope.Dispose();
            AssertTensorCounts(6, 2, 0, 4, 0, 0, 0);
            AssertPackedCounts(1, 0, 0, 1, 1, 0, 0);
            AssertTotalsCounts(7, 2, 0, 5, 1, 0, 0);
        }

        [Fact]
        public void DetachingAttachedIncrementsDetachedForSequenceButNotTensors()
        {
            using var ps = Create();
            using var scope = torch.NewDisposeScope();
            scope.Attach(ps);
            scope.Detach(ps);
            AssertTensorCounts(6, 2, 0, 0, 0, 0, 4);
            AssertPackedCounts(1, 0, 0, 0, 1, 1, 1);
            AssertTotalsCounts(7, 2, 0, 0, 1, 1, 5);
        }

        [Fact]
        public void DisposingScopeAfterDetachingDoesNothing()
        {
            using var ps = Create();
            var scope = torch.NewDisposeScope();
            scope.Attach(ps);
            scope.Detach(ps);
            scope.Dispose();
            AssertTensorCounts(6, 2, 0, 0, 0, 0, 4);
            AssertPackedCounts(1, 0, 0, 0, 1, 1, 1);
            AssertTotalsCounts(7, 2, 0, 0, 1, 1, 5);
        }
    }
}