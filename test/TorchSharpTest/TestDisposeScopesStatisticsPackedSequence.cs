using TorchSharp;
using Xunit;

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestDisposeScopesStatisticsPackedSequenceScoped : TestDisposeScopesStatisticsBase
    {

        //When reviewing these tests, refer first to TestDisposeScopesStatisticsTensor. The tests here are
        //essentially identical, except that the numbers all look weird because a PackedSequence is constructed
        //from Tensors, and contains Tensors that it detaches from any scope and manages itself.


        [Fact]
        public void CreatingIncrementsCreatedInScope()
        {
            using var scope = torch.NewDisposeScope();
            ResetStats();
            var ps = TestDisposeScopesStatisticsPackedSequenceUnscoped.Create();
            //CreatedInScope = 2 tensors (source data) + 4 internal PackedSequence tensors, + the PackedSequence
            //DetachedFromScope = the 4 internal PackedSequenceTensors
            AssertStatCounts(7, 0, 4, 0, 3);
        }

        [Fact]
        public void AttachingIncrementsNothing()
        {
            using var scope = torch.NewDisposeScope();
            var ps = TestDisposeScopesStatisticsPackedSequenceUnscoped.Create();
            ResetStats();
            scope.Attach(ps);
            AssertStatCounts(0, 0, 0, 0, 0);
        }

        [Fact]
        public void DetachingIncrementsDetached()
        {
            using var scope = torch.NewDisposeScope();
            var ps = TestDisposeScopesStatisticsPackedSequenceUnscoped.Create();
            ResetStats();
            scope.Detach(ps);
            AssertStatCounts(0, 0, 1, 0, -1);
        }

        [Fact]
        public void DisposingIncrementsDisposed()
        {
            using var scope = torch.NewDisposeScope();
            var ps = TestDisposeScopesStatisticsPackedSequenceUnscoped.Create();
            ResetStats();
            ps.Dispose();
            //DisposedInScope = The PackedSequence + the 4 internal tensors
            AssertStatCounts(0, 0, 0, 5, -5);
        }

        [Fact]
        public void DisposingScopeIncrementsDisposed()
        {
            var scope = torch.NewDisposeScope();
            var ps = TestDisposeScopesStatisticsPackedSequenceUnscoped.Create();
            ResetStats();
            scope.Dispose();
            //DisposedInScope = The PackedSequence and the 2 source data tensors. The internal tensors and not on the scope.
            AssertStatCounts(0, 0, 0, 3, -3);
        }

        [Fact]
        public void DisposingScopeAfterDetachingCountsOnlyTheSourceDataTensors()
        {
            var scope = torch.NewDisposeScope();
            var ps = TestDisposeScopesStatisticsPackedSequenceUnscoped.Create();
            scope.Detach(ps);
            ResetStats();
            scope.Dispose();
            //DisposedInScope just the two source data tensors.
            AssertStatCounts(0, 0, 0, 2, -2);
        }
    }
}