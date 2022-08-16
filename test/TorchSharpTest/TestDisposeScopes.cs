// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System.Collections.Generic;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace TorchSharp
{
#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
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
            DisposeScopeManager.Statistics.Reset();
            using (var scope1 = torch.NewDisposeScope()) {
                var a1 = 1.ToTensor(); // This one is caught
                var a2 = 2.ToTensor().DetatchFromDisposeScope(); // This one is lost
                var a3 = 3.ToTensor().MoveToOuterDisposeScope(); // This one is lost also
                using var a4 = 4.ToTensor(); // This one was manually disposed
                var disposables = scope1.DisposablesView;
                Assert.Multiple(
                    () => Assert.True(Contains(disposables, a1)),
                    () => Assert.False(Contains(disposables, a2)),
                    () => Assert.False(Contains(disposables, a3)),
                    () => Assert.True(Contains(disposables, a4))
                );
            }

            Assert.Multiple(
                () => Assert.Equal(0, DisposeScopeManager.Statistics.ThreadTotalLiveCount),
                // These numbers are higher than I expected them to be.
                // () => Assert.Equal(4, DisposeScopeManager.Statistics.CreatedInScopeCount),
                // () => Assert.Equal(2, DisposeScopeManager.Statistics.DisposedInScopeCount),
                () => Assert.Equal(2, DisposeScopeManager.Statistics.DetachedFromScopeCount),
                () => Assert.Equal(0, DisposeScopeManager.Statistics.CreatedOutsideScopeCount)
            );
            DisposeScopeManager.Statistics.Reset();
        }

        [Fact]
        public void DisposeEverythingButWorks()
        {
            using (var scope = torch.NewDisposeScope()) {
                var a1 = 1.ToTensor();
                var a2 = 2.ToTensor();
                var a3 = 3.ToTensor();
                var a4 = 4.ToTensor();

                var disposables = scope.DisposablesView;
                Assert.Multiple(
                    () => Assert.True(Contains(disposables, a1)),
                    () => Assert.True(Contains(disposables, a2)),
                    () => Assert.True(Contains(disposables, a3)),
                    () => Assert.True(Contains(disposables, a4))
                );
                scope.DisposeEverythingBut(a4);
                disposables = scope.DisposablesView;
                Assert.Multiple(
                    () => Assert.False(Contains(disposables, a1)),
                    () => Assert.False(Contains(disposables, a2)),
                    () => Assert.False(Contains(disposables, a3)),
                    () => Assert.True(Contains(disposables, a4))
                );
            }
            DisposeScopeManager.Statistics.Reset();
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
            DisposeScopeManager.Statistics.Reset();
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
            DisposeScopeManager.Statistics.Reset();
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
            DisposeScopeManager.Statistics.Reset();
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
            DisposeScopeManager.Statistics.Reset();
        }

        [Fact]
        public void DisposeScopesCanBeNestled()
        {
            DisposeScopeManager.Statistics.Reset();
            using (torch.NewDisposeScope()) {
                torch.Tensor data = torch.rand(10, 10);
                var t1 = data * data + data;

                var innerCount = DisposeScopeManager.Statistics.ThreadTotalLiveCount;
                using (torch.NewDisposeScope()) {
                    var t2 = data + data - t1;
                }

                // Inner scope was disposed
                Assert.Equal(DisposeScopeManager.Statistics.ThreadTotalLiveCount, innerCount);
            }

            // It was all disposed
            Assert.Equal(0, DisposeScopeManager.Statistics.ThreadTotalLiveCount);
            DisposeScopeManager.Statistics.Reset();
        }

        [Fact]
        public void DisposeScopeWorksForTestTraining1()
        {
            DisposeScopeManager.Statistics.Reset();
            using (var d = torch.NewDisposeScope()) {
                var testTraining = new TestTraining();
                testTraining.TestTraining1();
            }

            _testOutputHelper.WriteLine(
                $"Undisposed Tensors with DisposeScope: {DisposeScopeManager.Statistics.ThreadTotalLiveCount}");
            Assert.Equal(0, DisposeScopeManager.Statistics.ThreadTotalLiveCount);
            DisposeScopeManager.Statistics.Reset();
        }

        [Fact]
        public void DisposeScopeWorksForTestTrainingConv2d()
        {
            DisposeScopeManager.Statistics.Reset();
            using (var d = torch.NewDisposeScope()) {
                var testTraining = new TestTraining();
                testTraining.TestTrainingConv2d();
                _testOutputHelper.WriteLine($"Undisposed Tensors inside DisposeScope: {d.DisposablesCount}");
            }

            Assert.Equal(0, DisposeScopeManager.Statistics.ThreadTotalLiveCount);
            _testOutputHelper.WriteLine(
                $"Undisposed Tensors after DisposeScope: {DisposeScopeManager.Statistics.ThreadTotalLiveCount}");
            DisposeScopeManager.Statistics.Reset();
        }

        // Assert Contains causes problems!
        private bool Contains<T>(IReadOnlyList<T> list, T item) => list.Any(x => ReferenceEquals(x, item));
    }
}