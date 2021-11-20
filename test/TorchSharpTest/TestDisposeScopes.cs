// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System.Collections.Generic;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace TorchSharp
{
    public class TestDisposeScopes
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public TestDisposeScopes(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [Fact]
        public void MinimalDisposeWorks()
        {
            var preCount = DisposeScopeManager.ThreadTotalLiveCount;
            using (var scope1 = torch.NewDisposeScope()) {
                var a1 = 1.ToTensor(); // This one is caught
                var a2 = 2.ToTensor().DetatchFromDisposeScope(); // This one is lost
                var a3 = 3.ToTensor().MoveToOuterDisposeScope(); // This one is lost also
                using var a4 = 4.ToTensor(); // This one was manually disposed
            }

            Assert.Equal(preCount + 2, DisposeScopeManager.ThreadTotalLiveCount);
        }

        [Fact]
        public void StackedScopesWorksAsIntended()
        {
            torch.Tensor a1;
            using (var scope1 = torch.NewDisposeScope()) {
                a1 = 1.ToTensor();

                torch.Tensor a4;
                using (var scope2 = torch.NewDisposeScope()) {
                    // Moving to outer scope works
                    var a2 = 2f.ToTensor();
                    Assert.True(scope2.Contains(a2));
                    scope2.MoveToOuter(a2);
                    Assert.False(scope2.Contains(a2));
                    Assert.True(scope1.Contains(a1));

                    // Moving out of the scope alltogether works
                    var a6 = 6L.ToTensor();
                    scope2.Detach(a6);
                    // Assert.DoesNotContain Is broken!
                    Assert.False(scope2.Contains(a6));
                    Assert.False(scope1.Contains(a6));

                    var a3 = 3.ToTensor();
                    a4 = 4.ToTensor();
                    scope2.DisposeEverythingBut(a4);
                    Assert.True(a3.IsInvalid);
                    Assert.False(a4.IsInvalid);

                    var a5 = 5f.ToTensor();
                    Assert.True(scope2.Contains(a5));
                    a5.Dispose();
                    // a5 is invalid at this point
                    Assert.False(scope2.Contains(a5));
                }

                Assert.False(a1.IsInvalid);
                Assert.True(a4.IsInvalid);
            }

            Assert.True(a1.IsInvalid);
        }

        [Fact]
        public void TensorsAreDisposedCorrectly()
        {
            torch.Tensor data = torch.rand(10, 10);

            var disposables = new List<torch.Tensor>();
            using (var d = torch.NewDisposeScope()) {
                var newValue = data * data + data;
                disposables = d.DisposablesView.OfType<torch.Tensor>().ToList();
                foreach (var disposable in disposables) {
                    Assert.False(disposable.IsInvalid);
                }
            }

            // They were all disposed
            foreach (var disposable in disposables) {
                Assert.True(disposable.IsInvalid);
            }
        }

        [Fact]
        public void TensorsThatShouldNotBeDisposedArent()
        {
            torch.Tensor data = torch.rand(10, 10);
            torch.Tensor undisposed = null;
            using (var d = torch.NewDisposeScope()) {
                undisposed = d.MoveToOuter(data * data + data);
                Assert.False(undisposed.IsInvalid);
            }

            // One was kept
            Assert.False(undisposed.IsInvalid);
        }

        [Fact]
        public void TensorsCanBeDisposedInTheMiddleOfTheProcess()
        {
            torch.Tensor data = torch.rand(10, 10);
            torch.Tensor t1, t2;
            using (var d = torch.NewDisposeScope()) {
                t1 = data * data + data;
                t2 = data + data - t1;
                t2 = d.DisposeEverythingBut(t2);
                Assert.True(t1.IsInvalid);
                t2 = t2 + data;
            }

            Assert.True(t2.IsInvalid);
            Assert.False(data.IsInvalid);
        }

        [Fact]
        public void DisposeScopesCanBeNestled()
        {
            torch.Tensor data = torch.rand(10, 10);
            var preTotalCount = DisposeScopeManager.ThreadTotalLiveCount;

            using (torch.NewDisposeScope()) {
                var t1 = data * data + data;

                var innerCount = DisposeScopeManager.ThreadTotalLiveCount;
                using (torch.NewDisposeScope()) {
                    var t2 = data + data - t1;
                }

                // Inner scope was disposed
                Assert.Equal(DisposeScopeManager.ThreadTotalLiveCount, innerCount);
            }

            // It was all disposed
            Assert.Equal(DisposeScopeManager.ThreadTotalLiveCount, preTotalCount);
        }

        [Fact]
        public void DisposeScopeWorksForTestTraining1()
        {
            var count = DisposeScopeManager.ThreadTotalLiveCount;
            using (var d = torch.NewDisposeScope()) {
                var testTraining = new TestTraining();
                testTraining.TestTraining1();
            }

            _testOutputHelper.WriteLine(
                $"Undisposed Tensors with DisposeScope: {DisposeScopeManager.ThreadTotalLiveCount - count}");
            Assert.Equal(count, DisposeScopeManager.ThreadTotalLiveCount);
        }

        [Fact]
        public void DisposeScopeWorksForTestTrainingConv2d()
        {
            var count = DisposeScopeManager.ThreadTotalLiveCount;
            using (var d = torch.NewDisposeScope()) {
                var testTraining = new TestTraining();
                testTraining.TestTrainingConv2d();
                _testOutputHelper.WriteLine($"Undisposed Tensors inside DisposeScope: {d.DisposablesCount}");
            }

            Assert.Equal(count, DisposeScopeManager.ThreadTotalLiveCount);
            _testOutputHelper.WriteLine(
                $"Undisposed Tensors after DisposeScope: {DisposeScopeManager.ThreadTotalLiveCount - count}");
        }
    }
}