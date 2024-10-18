using TorchSharp;
using Xunit;

namespace TorchSharpTest;

[Collection("Sequential")]
public class TestDisposeScopesWithPackedSequence
{
    [Fact]
    public void PackSequencesMoveDisposeScope()
    {
        torch.nn.utils.rnn.PackedSequence packed_sequence;
        var otherScope = torch.NewDisposeScope();
        using (var outerScope = torch.NewDisposeScope()) {
            using (var innerScope = torch.NewDisposeScope()) {
                var sequences = make_sequence_tensors();
                packed_sequence = torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted: false);
                AssertPackedSequenceValid(packed_sequence);
                packed_sequence.MoveToOuterDisposeScope();
            }

            AssertPackedSequenceValid(packed_sequence);
            packed_sequence.MoveToOtherDisposeScope(otherScope);
        }

        AssertPackedSequenceValid(packed_sequence);
        otherScope.Dispose();
        Assert.True(packed_sequence.IsInvalid);
        Assert.True(packed_sequence.data.IsInvalid);
    }

    [Fact]
    public void PackedSequencesWorkWhenSorted()
    {
        var sequences = make_sequence_tensors();

        var scope = torch.NewDisposeScope();
        var packed_sequence = torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted: true);
        Assert.Equal(5, scope.DisposablesCount);
        Assert.False(packed_sequence.IsInvalid);
        Assert.False(packed_sequence.batch_sizes.IsInvalid);
        Assert.False(packed_sequence.data.IsInvalid);
        Assert.True(packed_sequence.sorted_indices.IsInvalid);
        Assert.True(packed_sequence.unsorted_indices.IsInvalid);

        scope.Dispose();
        Assert.True(packed_sequence.IsInvalid);
        Assert.True(packed_sequence.batch_sizes.IsInvalid);
        Assert.True(packed_sequence.data.IsInvalid);
        Assert.True(packed_sequence.sorted_indices.IsInvalid);
        Assert.True(packed_sequence.unsorted_indices.IsInvalid);
    }

    private static void AssertPackedSequenceValid(torch.nn.utils.rnn.PackedSequence packed_sequence)
    {
        Assert.False(packed_sequence.IsInvalid);
        Assert.False(packed_sequence.batch_sizes.IsInvalid);
        Assert.False(packed_sequence.data.IsInvalid);
        Assert.False(packed_sequence.sorted_indices.IsInvalid);
        Assert.False(packed_sequence.unsorted_indices.IsInvalid);
    }

    private static torch.Tensor[] make_sequence_tensors()
    {
        var sequences =
            new torch.Tensor[] { torch.tensor(new long[] { 1, 2, 3, 4 }), torch.tensor(new long[] { 5, 6 }), };
        return sequences;
    }
}
