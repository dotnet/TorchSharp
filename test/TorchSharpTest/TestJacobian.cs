using System.Linq;
using System.Collections.Generic;
using Xunit;

using static TorchSharp.torch;

namespace TorchSharp
{
    public class TestJacobian
    {
        [Fact]
        public void OneInputOneOutput()
        {
            torch.Tensor x1 = torch.tensor(new double[,] { { 1.0, 3.0 }, { 2.0, 4.0 } }, requires_grad: true);

            torch.Tensor jacFunc(torch.Tensor inputs)
            {
                return torch.sum(inputs * inputs);
            }

            torch.Tensor jacobian = torch.autograd.functional.jacobian(jacFunc, x1);

            Assert.Equal(new long[] { 2, 2 }, jacobian.shape);
            Assert.Equal(new double[] { 2.0, 6.0, 4.0, 8.0 }, jacobian.data<double>());

        }

        [Fact]
        public void OneInputMultipleOutputs()
        {
            torch.Tensor x1 = torch.tensor(new double[,] { { 1.0, 3.0 }, { 2.0, 4.0 } }, requires_grad: true);

            torch.Tensor[] jacFunc(torch.Tensor inputs)
            {
                return new torch.Tensor[] { torch.sum(inputs), torch.sum(inputs * inputs) };
            }

            torch.Tensor[] jacobian = torch.autograd.functional.jacobian(jacFunc, x1).ToArray();

            Assert.Equal(2, jacobian.Length);
            Assert.Equal(new long[] { 2, 2 }, jacobian[0].shape);
            Assert.Equal(new double[] { 1.0, 1.0, 1.0, 1.0 }, jacobian[0].data<double>());
            Assert.Equal(new long[] { 2, 2 }, jacobian[1].shape);
            Assert.Equal(new double[] { 2.0, 6.0, 4.0, 8.0 }, jacobian[1].data<double>());
        }

        [Fact]
        public void MultipleInputsOneOutput()
        {
            torch.Tensor x1 = torch.tensor(new double[,] { { 1.0, 3.0 }, { 2.0, 4.0 } }, requires_grad: true);
            torch.Tensor intercepts = torch.tensor(new double[] { 1.0, 5.0 }, requires_grad: true);

            torch.Tensor jacFunc(torch.Tensor[] inputs)
            {
                return torch.einsum("ij,j->i", inputs[0], inputs[1]);
            }

            torch.Tensor[] jacobian = torch.autograd.functional.jacobian(jacFunc, x1, intercepts).ToArray();

            // jacobian[0] should be:
            // [[[1, 5],
            // [0, 0]],
            // [[0, 0],
            // [1, 5]]]
            // And jacobian[1]:
            // [[1, 3],
            // [2, 4]]

            Assert.Equal(2, jacobian.Length);
            Assert.Equal(new long[] { 2, 2, 2 }, jacobian[0].shape);
            Assert.Equal(new double[] { 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0 }, jacobian[0].data<double>());
            Assert.Equal(new long[] { 2, 2 }, jacobian[1].shape);
            Assert.Equal(new double[] { 1.0, 3.0, 2.0, 4.0 }, jacobian[1].data<double>());
        }

        [Fact]
        public void MultipleInputsMultipleOutputs()
        {
            torch.Tensor x1 = torch.tensor(new double[,] { { 1.0, 3.0 }, { 2.0, 4.0 } }, requires_grad: true);
            torch.Tensor intercepts = torch.tensor(new double[] { 1.0, 5.0 }, requires_grad: true);

            torch.Tensor[] jacFunc(torch.Tensor[] inputs)
            {
                return new torch.Tensor[] { torch.sum(inputs[0] * inputs[0]), torch.einsum("ij,j->i", inputs[0], inputs[1]) };
            }

            torch.Tensor[] jacobian = torch.autograd.functional.jacobian(jacFunc, x1, intercepts).ToArray();

            Assert.Equal(4, jacobian.Length);
            Assert.Equal(new long[] { 2, 2 }, jacobian[0].shape);
            Assert.Equal(new double[] { 2.0, 6.0, 4.0, 8.0 }, jacobian[0].data<double>());
            Assert.Equal(new long[] { 2 }, jacobian[1].shape);
            Assert.Equal(new double[] { 0.0, 0.0 }, jacobian[1].data<double>());
            Assert.Equal(new long[] { 2, 2, 2 }, jacobian[2].shape);
            Assert.Equal(new double[] { 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0 }, jacobian[2].data<double>());
            Assert.Equal(new long[] { 2, 2 }, jacobian[3].shape);
            Assert.Equal(new double[] { 1.0, 3.0, 2.0, 4.0 }, jacobian[3].data<double>());

        }
    }
}