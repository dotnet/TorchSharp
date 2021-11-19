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
        public void StackedScopesWorksAsIntended()
        {
            torch.Tensor a1;
            using (var scope1 = torch.NewDisposeScope()) {
                a1 = 1.ToTensor();

                torch.Tensor a4;
                using (var scope2 = torch.NewDisposeScope()) {
                    // Moving to outer scope works
                    var a2 = 2f.ToTensor();
                    Assert.Contains(a2, scope2.Disposables);
                    scope2.Exclude(a2);
                    Assert.DoesNotContain(a2, scope2.Disposables);
                    Assert.Contains(a1, scope1.Disposables);

                    // Moving out of the scope alltogether works
                    var a6 = 6L.ToTensor();
                    scope2.ExcludeGlobally(a6);
                    // Assert.DoesNotContain Is broken!
                    var cont = scope2.Disposables.Contains(a6);
                    Assert.False(cont);
                    cont = scope1.Disposables.Contains(a6);
                    Assert.False(cont);

                    var a3 = 3.ToTensor();
                    a4 = 4.ToTensor();
                    scope2.DisposeEverythingBut(a4);
                    Assert.True(a3.IsDisposed);
                    Assert.False(a4.IsDisposed);

                    var a5 = 5f.ToTensor();
                    Assert.Contains(a5, scope2.Disposables);
                    a5.Dispose();
                    Assert.DoesNotContain(a5, scope2.Disposables);
                }

                Assert.False(a1.IsDisposed);
                Assert.True(a4.IsDisposed);
            }

            Assert.True(a1.IsDisposed);
        }

        [Fact]
        public void TensorsAreDisposedCorrectly()
        {
            torch.Tensor data = torch.rand(10, 10);

            var disposables = new List<torch.Tensor>();
            using (var d = torch.NewDisposeScope()) {
                var newValue = data * data + data;
                disposables = d.Disposables.OfType<torch.Tensor>().ToList();
                foreach (var disposable in disposables)
                {
                    Assert.False(disposable.IsDisposed);
                }
            }

            // They were all disposed
            foreach (var disposable in disposables)
            {
                Assert.True(disposable.IsDisposed);
            }
        }

        [Fact]
        public void TensorsThatShouldNotBeDisposedArent()
        {
            torch.Tensor data = torch.rand(10, 10);
            torch.Tensor undisposed = null;
            using (var d = torch.NewDisposeScope()) {
                undisposed = d.Exclude(data * data + data);
                Assert.False(undisposed.IsDisposed);
            }

            // One was kept
            Assert.False(undisposed.IsDisposed);
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
                Assert.True(t1.IsDisposed);
                t2 = t2 + data;
            }

            Assert.True(t2.IsDisposed);
            Assert.False(data.IsDisposed);
        }

        [Fact]
        public void DisposeScopesCanBeNestled()
        {
            torch.Tensor data = torch.rand(10, 10);
            var preTotalCount = DisposeScopeManager.Singleton.TotalCount;

            using (torch.NewDisposeScope()) {
                var t1 = data * data + data;

                var innerCount = DisposeScopeManager.Singleton.TotalCount;
                using (torch.NewDisposeScope()) {
                    var t2 = data + data - t1;
                }

                // Inner scope was disposed
                Assert.Equal(torch.Tensor.TotalCount, innerCount);
            }

            // It was all disposed
            Assert.Equal(torch.Tensor.TotalCount, preTotalCount);
        }

        [Fact]
        public void DisposeScopeWorksForTestTraining1()
        {
            var count = torch.Tensor.TotalCount;
            // No dispose scope
            var testTraining = new TestTraining();
            testTraining.TestTraining1();
            _testOutputHelper.WriteLine($"Undisposed Tensors without DisposeScope: {torch.Tensor.TotalCount - count}");

            count = torch.Tensor.TotalCount;
            using (var d = torch.NewDisposeScope()) {
                d.DontReportOnExternalDisposes = true;
                testTraining.TestTraining1();
            }

            _testOutputHelper.WriteLine($"Undisposed Tensors with DisposeScope: {torch.Tensor.TotalCount - count}");
            Assert.Equal(torch.Tensor.TotalCount, count);
        }

        [Fact]
        public void DisposeScopeWorksForTestTrainingConv2d()
        {
            var count = torch.Tensor.TotalCount;
            // No dispose scope
            var testTraining = new TestTraining();
            using (var d = torch.NewDisposeScope()) {
                d.DontReportOnExternalDisposes = true;
                testTraining.TestTrainingConv2d();
                _testOutputHelper.WriteLine($"Undisposed Tensors inside DisposeScope: {d.Disposables.Count}");
            }

            _testOutputHelper.WriteLine($"Undisposed Tensors after DisposeScope: {torch.Tensor.TotalCount - count}");
        }
    }
}