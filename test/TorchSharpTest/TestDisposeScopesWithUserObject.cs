using TorchSharp;
using Xunit;

namespace TorchSharpTest;

[Collection("Sequential")]
public class TestDisposeScopesWithUserObject
{
    [Fact]
    public void UserObjectCanParticipateInScopeSystem()
    {
        var scope = torch.NewDisposeScope();
        var custom = new CustomScopedObject();
        Assert.False(custom.IsInvalid);
        Assert.Equal(0, scope.DisposablesCount);

        scope.Attach(custom);
        Assert.Equal(1, scope.DisposablesCount);

        scope.Dispose();
        Assert.True(custom.IsInvalid);
    }

    private class CustomScopedObject : IDisposeScopeClient
    {
        public void Dispose()
        {
            IsInvalid = true;
        }

        public DisposeScope OwningDisposeScope { get; set; }
        public bool IsInvalid { get; private set; } = false;
    }
}