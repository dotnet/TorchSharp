// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System.Reflection;
using Xunit;

#nullable enable
namespace TorchSharp
{
    public class TestTorchHub
    {
        [Fact]
        public void TestProgressBar()
        {
            var type = typeof(torch.hub);
            foreach (var method in type.GetMethods(BindingFlags.NonPublic | BindingFlags.Static)) {
                if (method.Name == "_create_progress_bar") {
                    var result = (IProgressBar?)method.Invoke(null, new object[] { false });
                    Assert.NotNull(result);
                    Assert.Equal("TorchSharp.ConsoleProgressBar", result.GetType().FullName);
                    Assert.Equal(0, result.Value);
                    Assert.Null(result.Maximum);
                    result.Value = 100;
                    result.Maximum = 50;
                    Assert.Equal(50, result.Value);
                }
            }
        }
    }
}
