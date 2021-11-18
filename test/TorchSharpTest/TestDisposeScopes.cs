using System.Threading;
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
        public void TensorsAreDisposedCorrectly()
        {
            torch.Tensor data = torch.rand(10, 10);
            var preTotalCount = torch.Tensor.TotalCount;
            using (var d = data.NewDisposeScope()) {
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
            using (var d = data.NewDisposeScope()) {
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
            using (var d = data.NewDisposeScope()) {
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

            using (data.NewDisposeScope()) {
                var t1 = data * data + data;

                var innerCount = torch.Tensor.TotalCount;
                using (data.NewDisposeScope()) {
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
            using (torch.CPU.NewDisposeScope()) {
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
            using (var d = torch.CPU.NewDisposeScope()) {
                testTraining.TestTrainingConv2d();
                _testOutputHelper.WriteLine($"Undisposed Tensors inside DisposeScope: {d.Tensors.Count}");
            }

            _testOutputHelper.WriteLine($"Undisposed Tensors after DisposeScope: {torch.Tensor.TotalCount - count}");
        }
    }
}