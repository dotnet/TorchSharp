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
        public void MemoryManagementExample()
        {
            // This test replicates the code from the memory management documentation at TorchSharp
            torch.Tensor sum;
            _testOutputHelper.WriteLine($"Total tensors: {torch.Tensor.TotalCount}"); // Writes 0
            using (var d = torch.NewDisposeScope()) {
                var predictedLabels = torch.rand(5, 5);
                var labels = torch.tensor(new int[] { 1, 0, 1, 0, 1 });
                var am = predictedLabels.argmax(1);
                var eq = am == labels;

                _testOutputHelper.WriteLine($"Total tensors temporary dispose: {torch.Tensor.TotalCount}"); // Writes 4

                // If the process is long, you can dispose mid stream;
                d.DisposeEverythingBut(eq);

                // This last value is excluded from the DisposeScope, so it can be used outside.
                sum = d.Exclude(eq.sum());
                _testOutputHelper.WriteLine($"Total tensors before dispose: {torch.Tensor.TotalCount}"); // Writes 2
            }

            _testOutputHelper.WriteLine($"Total tensors after dispose: {torch.Tensor.TotalCount}"); // Writes 1
            _testOutputHelper.WriteLine($"Tensors disposed: {DisposeScopeManager.DisposedTensorCount}"); // Writes 4
            _testOutputHelper.WriteLine($"sum:{sum.ToSingle()}");
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
            var preTotalCount = torch.Tensor.TotalCount;
            using (var d = torch.NewDisposeScope()) {
                var newValue = data * data + data;
                Assert.True(torch.Tensor.TotalCount > preTotalCount);
            }

            // They were all disposed
            Assert.Equal(torch.Tensor.TotalCount, preTotalCount);
        }

        [Fact]
        public void TensorsThatShouldNotBeDisposedArent()
        {
            torch.Tensor data = torch.rand(10, 10);
            var preTotalCount = torch.Tensor.TotalCount;
            using (var d = torch.NewDisposeScope()) {
                var newValue = d.Exclude(data * data + data);
                Assert.True(torch.Tensor.TotalCount > preTotalCount);
            }

            // One was kept
            Assert.Equal(torch.Tensor.TotalCount, preTotalCount + 1);
        }

        [Fact]
        public void TensorsCanBeDisposedInTheMiddleOfTheProcess()
        {
            torch.Tensor data = torch.rand(10, 10);
            var preTotalCount = torch.Tensor.TotalCount;
            using (var d = torch.NewDisposeScope()) {
                var t1 = data * data + data;
                var t2 = data + data - t1;
                Assert.True(torch.Tensor.TotalCount > preTotalCount + 1);
                t2 = d.DisposeEverythingBut(t2);
                Assert.Equal(torch.Tensor.TotalCount, preTotalCount + 1);
                t2 = t2 + data;
            }

            // It was all disposed
            Assert.Equal(torch.Tensor.TotalCount, preTotalCount);
        }

        [Fact]
        public void DisposeScopesCanBeNestled()
        {
            torch.Tensor data = torch.rand(10, 10);
            var preTotalCount = torch.Tensor.TotalCount;

            using (torch.NewDisposeScope()) {
                var t1 = data * data + data;

                var innerCount = torch.Tensor.TotalCount;
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