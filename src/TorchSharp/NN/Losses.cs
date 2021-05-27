// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

#nullable enable

namespace TorchSharp.NN
{
    public delegate TorchTensor Loss(TorchTensor source, TorchTensor target);

    /// <summary>
    /// Class maintaing the supported loss functions.
    /// </summary>
    public static partial class Functions
    {

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_cross_entropy(IntPtr srct, IntPtr trgt, IntPtr wgt, long ignore_index, bool hasII, long reduction);

        /// <summary>
        /// This criterion combines log_softmax and nll_loss in a single function.
        /// </summary>
        /// <param name="weight">A manual rescaling weight if provided it’s repeated to match input tensor shape</param>
        /// <param name="ignore_index">Specifies a target value that is ignored and does not contribute to the input gradient.</param>
        /// <param name="reduction">Specifies the reduction to apply to the output</param>
        /// <returns></returns>
        public static Loss cross_entropy_loss(TorchTensor? weight = null, long? ignore_index = null, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                var ii = ignore_index.HasValue ? ignore_index.Value : -100;
                var res = THSNN_cross_entropy(src.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, ii, ignore_index.HasValue, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_binary_cross_entropy(IntPtr srct, IntPtr trgt, IntPtr wgt, long reduction);

        /// <summary>
        /// Measures the Binary Cross Entropy between the target and the output.
        /// </summary>
        /// <param name="weight">A manual rescaling weight if provided it’s repeated to match input tensor shape</param>
        /// <param name="reduction">Specifies the reduction to apply to the output</param>
        /// <returns></returns>
        public static Loss binary_cross_entropy_loss(TorchTensor? weight = null, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                var res = THSNN_binary_cross_entropy(src.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_binary_cross_entropy_with_logits(IntPtr srct, IntPtr trgt, IntPtr wgt, long reduction, IntPtr posWeights);

        /// <summary>
        /// Measures Binary Cross Entropy between target and output logits.
        /// </summary>
        /// <param name="weight">A manual rescaling weight if provided it’s repeated to match input tensor shape</param>
        /// <param name="reduction">Specifies the reduction to apply to the output</param>
        /// <param name="posWeights">A weight of positive examples. Must be a vector with length equal to the number of classes.</param>
        /// <returns></returns>
        public static Loss binary_cross_entropy_with_logits_loss(TorchTensor? weight = null, Reduction reduction = Reduction.Mean, TorchTensor? posWeights = null)
        {
            return (TorchTensor src, TorchTensor target) => {
                var res = THSNN_binary_cross_entropy_with_logits(src.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, (long)reduction, posWeights?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_cosine_embedding_loss(IntPtr input1, IntPtr input2, IntPtr trgt, double margin, long reduction);

        /// <summary>
        /// Measures the loss given two input tensor and a lable tensor with values 1 or -1.
        ///
        /// See: https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html#torch.nn.CosineEmbeddingLoss
        /// </summary>
        /// <param name="margin"> Should be a number from -1 to 1, 0 to 0.5 is suggested</param>
        /// <param name="reduction">Specifies the reduction to apply to the output</param>
        /// <returns></returns>
        public static TwoInputLoss cosine_embedding_loss(double margin = 0.0, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor input1, TorchTensor input2, TorchTensor target) => {
                var res = THSNN_cosine_embedding_loss(input1.Handle, input2.Handle, target.Handle, margin, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        public delegate TorchTensor TwoInputLoss(TorchTensor input1, TorchTensor input2, TorchTensor target);


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ctc_loss(IntPtr log_probs, IntPtr targets, IntPtr input_lengths, IntPtr target_lengths, long blank, bool zero_infinity, long reduction);

        /// <summary>
        /// The Connectionist Temporal Classification loss.
        ///
        /// Calculates loss between a continuous (unsegmented) time series and a target sequence.
        /// CTCLoss sums over the probability of possible alignments of input to target, producing a
        /// loss value which is differentiable with respect to each input node. The alignment of input to
        /// target is assumed to be “many-to-one”, which limits the length of the target sequence such that
        /// it must be less than the input length.
        /// </summary>
        /// <returns></returns>
        public static CTCLoss ctc_loss(long blank = 0, bool zeroInfinity = false, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor log_probs, TorchTensor targets, TorchTensor input_lengths, TorchTensor target_lengths) => {
                var res = THSNN_ctc_loss(log_probs.Handle, targets.Handle, input_lengths.Handle, target_lengths.Handle, blank, zeroInfinity, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        public delegate TorchTensor CTCLoss(TorchTensor log_probs, TorchTensor targets, TorchTensor input_lengths, TorchTensor target_lengths);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_hinge_embedding_loss(IntPtr input, IntPtr trgt, double margin, long reduction);

        /// <summary>
        /// Measures the loss given an input tensor x and a labels tensor y (containing 1 or -1).
        ///
        /// See: https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html#torch.nn.HingeEmbeddingLoss
        /// </summary>
        /// <param name="margin"> Should be a number from -1 to 1, 0 to 0.5 is suggested</param>
        /// <param name="reduction">Specifies the reduction to apply to the output</param>
        /// <returns></returns>
        public static Loss hinge_embedding_loss(double margin = 0.0, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor input, TorchTensor target) => {
                var res = THSNN_hinge_embedding_loss(input.Handle, target.Handle, margin, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_margin_ranking_loss(IntPtr input1, IntPtr input2, IntPtr target, double margin, long reduction);

        public static TwoInputLoss margin_ranking_loss(double margin = 0, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor input1, TorchTensor input2, TorchTensor target) => {
                var res = THSNN_margin_ranking_loss(input1.Handle, input2.Handle, target.Handle, margin, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_multilabel_margin_loss(IntPtr input, IntPtr target, long reduction);

        public static Loss multilabel_margin_loss(Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor input, TorchTensor target) => {
                var res = THSNN_multilabel_margin_loss(input.Handle, target.Handle, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_multilabel_soft_margin_loss(IntPtr input, IntPtr target, IntPtr weight, long reduction);

        public static Loss multilabel_soft_margin_loss(TorchTensor? weight = null, Reduction reduction = Reduction.Mean)
        {
            IntPtr h = (weight is null) ? IntPtr.Zero : weight.Handle;

            return (TorchTensor input, TorchTensor target) => {
                var res = THSNN_multilabel_soft_margin_loss(input.Handle, target.Handle, h, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_multi_margin_loss(IntPtr input, IntPtr target, long p, double margin, IntPtr weight, long reduction);

        public static Loss multi_margin_loss(int p = 1, double margin = 1.0, TorchTensor? weight = null, Reduction reduction = Reduction.Mean)
        {
            IntPtr h = (weight is null) ? IntPtr.Zero : weight.Handle;

            return (TorchTensor input, TorchTensor target) => {
                var res = THSNN_multi_margin_loss(input.Handle, target.Handle, p, margin, h, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_mse_loss(IntPtr srct, IntPtr trgt, long reduction);

        /// <summary>
        /// Measures the element-wise mean squared error.
        /// </summary>
        /// <param name="reduction">Specifies the reduction to apply to the output</param>
        /// <returns></returns>
        public static Loss mse_loss(Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                var res = THSNN_mse_loss(src.Handle, target.Handle, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_l1_loss(IntPtr srct, IntPtr trgt, long reduction);

        /// <summary>
        /// Function that takes the mean element-wise absolute value difference.
        /// </summary>
        /// <param name="reduction">Specifies the reduction to apply to the output</param>
        /// <returns></returns>
        public static Loss l1_loss(Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                var res = THSNN_l1_loss(src.Handle, target.Handle, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_nll_loss(IntPtr srct, IntPtr trgt, IntPtr wgt, long reduction);

        /// <summary>
        /// The negative log likelihood loss.
        /// </summary>
        /// <param name="weight">A manual rescaling weight if provided it’s repeated to match input tensor shape</param>
        /// <param name="reduction">Specifies the reduction to apply to the output</param>
        /// <returns></returns>
        public static Loss nll_loss(TorchTensor? weight = null, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                var res = THSNN_nll_loss(src.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_poisson_loss(IntPtr srct, IntPtr trgt, bool logInput, bool full, float eps, long reduction);

        /// <summary>
        /// Poisson negative log likelihood loss.
        /// </summary>
        /// <param name="logInput"></param>
        /// <param name="full">Whether to compute full loss, i. e. to add the Stirling approximation term.</param>
        /// <param name="eps">Small value to avoid evaluation of log(0)</param>
        /// <param name="reduction">Specifies the reduction to apply to the output</param>
        /// <returns></returns>
        public static Loss poisson_loss(bool logInput = true, bool full = false, float eps = 1e-8f, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                var res = THSNN_poisson_loss(src.Handle, target.Handle, logInput, full, eps, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_kl_div_loss(IntPtr input, IntPtr target, long reduction, bool logTarget);

        /// <summary>
        /// The Kullback-Leibler divergence Loss
        /// </summary>
        /// <param name="logTarget">A flag indicating whether target is passed in the log space.</param>
        /// <param name="reduction">Specifies the reduction to apply to the output</param>
        /// <returns></returns>
        public static Loss kl_div_loss(bool logTarget = true, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                var res = THSNN_kl_div_loss(src.Handle, target.Handle, (long)reduction, logTarget);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_smooth_l1_loss(IntPtr srct, IntPtr trgt, long reduction, double beta);

        /// <summary>
        /// Function that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise.
        /// </summary>
        /// <param name="logInput"></param>
        /// <param name="reduction">Specifies the reduction to apply to the output</param>
        /// <returns></returns>
        public static Loss smooth_l1_loss(bool logInput = true, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                // Currently, the 'beta' parameter is being ignored by the native layer, so we just pass the default.
                var res = THSNN_smooth_l1_loss(src.Handle, target.Handle, (long)reduction, 1.0);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_soft_margin_loss(IntPtr srct, IntPtr trgt, long reduction);

        /// <summary>
        /// Optimizes a two-class classification logistic loss between input tensor xx and target tensor yy (containing 1 or -1).
        /// </summary>
        /// <param name="reduction">Specifies the reduction to apply to the output</param>
        /// <returns></returns>
        public static Loss soft_margin_loss(Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                var res = THSNN_soft_margin_loss(src.Handle, target.Handle, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_triplet_margin_loss(IntPtr anchor, IntPtr positive, IntPtr negative, double margin, long p, double eps, bool swap, long reduction);

        public static TripletMarginLoss triplet_margin_loss(double margin = 1.0, long p = 2, double eps = 1e-06, bool swap = false, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor anchor, TorchTensor positive, TorchTensor negative) => {
                var res = THSNN_triplet_margin_loss(anchor.Handle, positive.Handle, negative.Handle, margin, p, eps, swap, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_triplet_margin_with_distance_loss(IntPtr anchor, IntPtr positive, IntPtr negative, DistanceFunctionNative? distance_function, double margin, bool swap, long reduction);

        public static TripletMarginLoss triplet_margin_with_distance_loss(Func<TorchTensor, TorchTensor, TorchTensor>? distance = null, double margin = 1.0, bool swap = false, Reduction reduction = Reduction.Mean)
        {
            if (distance != null) {
                return (TorchTensor anchor, TorchTensor positive, TorchTensor negative) => {
                    DistanceFunctionNative func = (IntPtr x, IntPtr y) => {
                        using (var x1 = new TorchTensor(x))
                        using (var y1 = new TorchTensor(y))
                            return distance(x1, y1).Handle;
                    };
                    var res = THSNN_triplet_margin_with_distance_loss(anchor.Handle, positive.Handle, negative.Handle, func, margin, swap, (long)reduction);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                };
            } else {
                return (TorchTensor anchor, TorchTensor positive, TorchTensor negative) => {
                    var res = THSNN_triplet_margin_with_distance_loss(anchor.Handle, positive.Handle, negative.Handle, null, margin, swap, (long)reduction);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                };
            }

        }

        public delegate TorchTensor TripletMarginLoss(TorchTensor anchor, TorchTensor positive, TorchTensor negative);

        /// <summary>
        /// Gaussian negative log likelihood loss.
        /// </summary>
        /// <param name="full">Include the constant term in the loss calculation</param>
        /// <param name="eps">Value used to clamp var (see note below), for stability.</param>
        /// <param name="reduction">Specifies the reduction to apply to the output</param>
        /// <returns></returns>
        public static GaussianNLLLoss gaussian_nll_loss(bool full = false, float eps = 1e-8f, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor input, TorchTensor target, TorchTensor variance) => {
                input = input.view(input.shape[0], -1);
                target = target.view(target.shape[0], -1);
                if (target.shape == input.shape) throw new ArgumentException("input and target must have the same shape");

                variance = variance.view(target.shape[0], -1);
                if (variance.shape[1] != input.shape[1] && variance.shape[1] != 1) throw new ArgumentException("variance has the wrong shape");

                if ((variance < 0).any().DataItem<bool>()) throw new ArgumentException("variance has negative entry/entries");

                variance = variance.clone().max(Float32Tensor.from(eps));

                var loss = 0.5 * (variance.log() + (input - target).square() / variance).view(input.shape[0], -1).sum(dimensions: new long[] { 1 });

                if (full) {
                    loss = loss + 0.5 * input.shape[1] * MathF.Log(2 * MathF.PI);
                }

                return (reduction == Reduction.Mean) ? loss.mean() : (reduction == Reduction.Sum) ? loss.sum() : loss;
            };
        }

        public delegate TorchTensor GaussianNLLLoss(TorchTensor source, TorchTensor target, TorchTensor variance);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr DistanceFunctionNative(IntPtr x, IntPtr y);


    }
    public enum Reduction : long
    {
        None = 0,
        Mean = 1,
        Sum = 2
    }
}
