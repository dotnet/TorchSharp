using System;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class nn
        {
            public static class init
            {
                public enum NonlinearityType
                {
                    Linear = 0,
                    Conv1D = 1,
                    Conv2D = 2,
                    Conv3D = 3,
                    ConvTranspose1D = 4,
                    ConvTranspose2D = 5,
                    ConvTranspose3D = 6,
                    Sigmoid = 7,
                    Tanh = 8,
                    ReLU = 9,
                    LeakyReLU = 10
                }

                public enum FanInOut
                {
                    FanIn = 0,
                    FanOut = 1
                }

                /// <summary>
                /// Return the recommended gain value for the given nonlinearity function.
                /// </summary>
                public static double calculate_gain(NonlinearityType nonlinearity, double param = 0.0)
                {
                    return THSInit_calculate_gain((long)nonlinearity, param);
                }

                /// <summary>
                /// Fills the input Tensor with the value 1
                /// </summary>
                public static Tensor ones_(Tensor tensor)
                {
                    THSInit_ones_(tensor.Handle);
                    torch.CheckForErrors();
                    return tensor;
                }

                /// <summary>
                /// Fills the input Tensor with the value 0
                /// </summary>
                public static Tensor zeros_(Tensor tensor)
                {
                    THSInit_zeros_(tensor.Handle);
                    torch.CheckForErrors();
                    return tensor;
                }

                /// <summary>
                /// Fills the {3, 4, 5}-dimensional input Tensor with the Dirac delta function.
                /// Preserves the identity of the inputs in Convolutional layers, where as many input channels are preserved as possible.
                /// </summary>
                public static Tensor dirac_(Tensor tensor)
                {
                    THSInit_dirac_(tensor.Handle);
                    torch.CheckForErrors();
                    return tensor;
                }

                /// <summary>
                /// Fills the 2-dimensional input Tensor with the identity matrix.
                /// Preserves the identity of the inputs in Linear layers, where as many inputs are preserved as possible.
                /// </summary>
                public static Tensor eye_(Tensor tensor)
                {
                    THSInit_eye_(tensor.Handle);
                    torch.CheckForErrors();
                    return tensor;
                }

                /// <summary>
                /// Fills the input Tensor with the value 'val'
                /// </summary>
                public static Tensor constant_(Tensor tensor, Scalar val)
                {
                    THSInit_constant_(tensor.Handle, val.Handle);
                    torch.CheckForErrors();
                    return tensor;
                }

                /// <summary>
                /// Fills the input Tensor with values drawn from the uniform distribution
                /// </summary>
                public static Tensor uniform_(Tensor tensor, double low = 0, double high = 1, Generator generator = null)
                {
                    THSInit_uniform_(tensor.Handle, low, high);
                    torch.CheckForErrors();
                    return tensor;
                }

                /// <summary>
                /// Fills the input Tensor with values drawn from the normal distribution
                /// </summary>
                public static Tensor normal_(Tensor tensor, double mean = 0, double std = 1, Generator generator = null)
                {
                    THSInit_normal_(tensor.Handle, mean, std);
                    torch.CheckForErrors();
                    return tensor;
                }

                /// <summary>
                /// Fills the input Tensor with values drawn from a truncated normal distribution.
                /// </summary>
                /// <param name="tensor">Input tensor</param>
                /// <param name="mean">The mean of the normal distribution</param>
                /// <param name="std">The standard deviation of the normal distribution</param>
                /// <param name="a">The minimum cutoff value</param>
                /// <param name="b">The maximum cutoff value</param>
                /// <param name="generator">An optional random number generator.</param>
                /// <returns></returns>
                public static Tensor trunc_normal_(Tensor tensor, double mean = 0.0, double std = 1.0, double a = -2.0, double b = 2.0, Generator generator = null)
                {
                    using (torch.no_grad()) {
                        THSInit_trunc_normal_(tensor.Handle, mean, std, a, b);
                        torch.CheckForErrors();
                        return tensor;
                    }
                }

                /// <summary>
                /// Fills the input Tensor with a (semi) orthogonal matrix, as described in 'Exact solutions to the nonlinear dynamics of learning in deep linear neural networks'
                /// </summary>
                public static Tensor orthogonal_(Tensor tensor, double gain = 1.0)
                {
                    THSInit_orthogonal_(tensor.Handle, gain);
                    torch.CheckForErrors();
                    return tensor;
                }

                /// <summary>
                /// Fills the 2D input Tensor as a sparse matrix, where the non-zero elements will be drawn from the normal distribution N(0,std)
                /// </summary>
                public static Tensor sparse_(Tensor tensor, double sparsity, double std = 0.01)
                {
                    THSInit_sparse_(tensor.Handle, sparsity, std);
                    torch.CheckForErrors();
                    return tensor;
                }

                /// <summary>
                /// Fills the input Tensor with values according to the method described in 'Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification'
                /// </summary>
                public static Tensor kaiming_uniform_(Tensor tensor, double a = 0, FanInOut mode = FanInOut.FanIn, NonlinearityType nonlinearity = NonlinearityType.LeakyReLU)
                {
                    THSInit_kaiming_uniform_(tensor.Handle, a, (long)mode, (long)nonlinearity);
                    torch.CheckForErrors();
                    return tensor;
                }

                /// <summary>
                /// Fills the input Tensor with values according to the method described in 'Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification'
                /// </summary>
                public static Tensor kaiming_normal_(Tensor tensor, double a = 0, FanInOut mode = FanInOut.FanIn, NonlinearityType nonlinearity = NonlinearityType.LeakyReLU)
                {
                    THSInit_kaiming_normal_(tensor.Handle, a, (long)mode, (long)nonlinearity);
                    torch.CheckForErrors();
                    return tensor;
                }

                /// <summary>
                /// Fills the input Tensor with values according to the method described in 'Understanding the difficulty of training deep feedforward neural networks'
                /// </summary>
                public static Tensor xavier_uniform_(Tensor tensor, double gain = 1.0)
                {
                    THSInit_xavier_uniform_(tensor.Handle, gain);
                    torch.CheckForErrors();
                    return tensor;
                }

                /// <summary>
                /// Fills the input Tensor with values according to the method described in 'Understanding the difficulty of training deep feedforward neural networks'
                /// </summary>
                public static Tensor glorot_uniform_(Tensor tensor, double gain = 1.0) => xavier_uniform_(tensor, gain);

                /// <summary>
                /// Fills the input Tensor with values according to the method described in 'Understanding the difficulty of training deep feedforward neural networks'
                /// </summary>
                public static Tensor xavier_normal_(Tensor tensor, double gain = 1.0)
                {
                    THSInit_xavier_uniform_(tensor.Handle, gain);
                    torch.CheckForErrors();
                    return tensor;
                }

                /// <summary>
                /// Fills the input Tensor with values according to the method described in 'Understanding the difficulty of training deep feedforward neural networks'
                /// </summary>
                public static Tensor glorot_normal_(Tensor tensor, double gain = 1.0) => xavier_normal_(tensor, gain);

                public static (long fanIn, long fanOut) CalculateFanInAndFanOut(Tensor tensor)
                {
                    var dimensions = tensor.Dimensions;

                    if (dimensions < 2) {
                        throw new ArgumentException("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions");
                    }

                    var shape = tensor.shape;
                    // Linear
                    if (dimensions == 2) {
                        return (shape[1], shape[0]);
                    } else {
                        var numInputFMaps = tensor.shape[1];
                        var numOutputFMaps = tensor.shape[0];
                        var receptiveFieldSize = tensor[0, 0].NumberOfElements;

                        return (numInputFMaps * receptiveFieldSize, numOutputFMaps * receptiveFieldSize);
                    }
                }
            }
        }
    }
}
