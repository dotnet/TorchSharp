using System.Reflection;
using TorchSharp;
using Xunit;

namespace TorchSharpTest;

public class TestDisposeScopesPackedSequence
{
    [Fact]
    public void MoveDisposeScope()
    {
        var sequences = CreateTestSequences();
        torch.nn.utils.rnn.PackedSequence packed_sequence;
        var otherScope = torch.NewDisposeScope();
        using (torch.NewDisposeScope())
        {
            using (torch.NewDisposeScope())
            {
                packed_sequence = torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted: false);
                AssertPackedSequenceValid(packed_sequence);

                packed_sequence.MoveToOuterDisposeScope();
            }
            AssertPackedSequenceValid(packed_sequence);

            packed_sequence.MoveToOtherDisposeScope(otherScope);
        }

        AssertPackedSequenceValid(packed_sequence);
        otherScope.Dispose();

        Assert.True(GetPackedSequenceIsInvalid(packed_sequence));
        Assert.True(packed_sequence.data.IsInvalid);
    }

    [Fact]
    public void DisposablesValidityWhenNotSorted()
    {
        var sequences = CreateTestSequences();
        using var scope = torch.NewDisposeScope();
        var packed = torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted: false);
        Assert.Equal(1, scope.DisposablesCount);
        AssertPackedSequenceValid(packed);
    }

    [Fact]
    public void DisposablesValidityWhenSorted()
    {
        var sequences = CreateTestSequences();
        using var scope = torch.NewDisposeScope();
        var packed = torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted: true);
        Assert.Equal(1, scope.DisposablesCount);
        Assert.False(GetPackedSequenceIsInvalid(packed));
        Assert.False(packed.batch_sizes.IsInvalid);
        Assert.False(packed.data.IsInvalid);
        Assert.True(packed.sorted_indices.IsInvalid);
        Assert.True(packed.unsorted_indices.IsInvalid);
    }

    [Fact]
    public void DisposeScopeStatistics()
    {
        DisposeScopeManager.Statistics.Reset();
        AssertStatCounts(0, 0, 0, 0, 0);
        var sequences = CreateTestSequences();
        AssertStatCounts(0, 2, 0, 0, 0);
        var outOfScope = torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted: true);
        AssertStatCounts(0, 7, 0, 0, 0);
        using var scope = torch.NewDisposeScope();
        AssertStatCounts(0, 7, 0, 0, 0);

        var inScope = torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted: true);
        AssertStatCounts(5, 7, 4, 0, 1);

        scope.Attach(outOfScope);
        //Possible subtle bug. When attaching an object that isn't owned by any scope, the count subtracts.
        AssertStatCounts( 5, 7, 3, 0, 2);

        scope.Detach(inScope);
        AssertStatCounts( 5, 7, 4, 0, 1);

        outOfScope.Dispose();
        AssertStatCounts( 5, 7, 4, 5, -4);

    }

    private static void AssertStatCounts(long createdInScope, long createdOutsideScope, long detachedFrom, long disposedIn, long threadTotalLive)
    {
        var stats = DisposeScopeManager.Statistics;
        Assert.Equal(createdInScope, stats.CreatedInScopeCount);
        Assert.Equal(createdOutsideScope, stats.CreatedOutsideScopeCount);
        Assert.Equal(detachedFrom, stats.DetachedFromScopeCount);
        Assert.Equal(disposedIn, stats.DisposedInScopeCount);
        Assert.Equal(threadTotalLive, stats.ThreadTotalLiveCount);
    }

    private static torch.Tensor[] CreateTestSequences()
    {
        return new[]
        {
            torch.tensor(new long[] { 1, 2, 3, 4 }),
            torch.tensor(new long[] { 5, 6 }),
        };
    }

    private static void AssertPackedSequenceValid(torch.nn.utils.rnn.PackedSequence packed_sequence)
    {
        Assert.False(GetPackedSequenceIsInvalid(packed_sequence));
        Assert.False(packed_sequence.batch_sizes.IsInvalid);
        Assert.False(packed_sequence.data.IsInvalid);
        Assert.False(packed_sequence.sorted_indices.IsInvalid);
        Assert.False(packed_sequence.unsorted_indices.IsInvalid);
    }

    private static bool GetPackedSequenceIsInvalid(torch.nn.utils.rnn.PackedSequence packed_sequence)
    {
        //HACK: reflection to avoid exposing internal method IsInvalid in API.
        var getter = typeof(torch.nn.utils.rnn.PackedSequence).GetProperty("IsInvalid", BindingFlags.Instance | BindingFlags.NonPublic)!;
        return (bool)getter.GetValue(packed_sequence)!;
    }
}