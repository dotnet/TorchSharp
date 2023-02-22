// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.PInvoke.LibTorchSharp;

#nullable enable

namespace TorchSharp
{
    using Modules;

    public abstract class Loss<T1, T2, TResult> : nn.Module<T1, T2, TResult>
    {
        public Loss(torch.nn.Reduction reduction = nn.Reduction.Mean) : base(nameof(Loss<T1, T2, TResult>))
        {
            this.reduction = reduction;
        }

        public torch.nn.Reduction reduction { get; }
    }

    public abstract class Loss<T1, T2, T3, TResult> : nn.Module<T1, T2, T3, TResult>
    {
        public Loss(torch.nn.Reduction reduction = nn.Reduction.Mean) : base(nameof(Loss<T1, T2, T3, TResult>))
        {
            this.reduction = reduction;
        }

        public torch.nn.Reduction reduction { get; }
    }
    public abstract class Loss<T1, T2, T3, T4, TResult> : nn.Module<T1, T2, T3, T4, TResult>
    {
        public Loss(torch.nn.Reduction reduction = nn.Reduction.Mean) : base(nameof(Loss<T1, T2, T3, T4, TResult>))
        {
            this.reduction = reduction;
        }

        public torch.nn.Reduction reduction { get; }
    }

    public abstract class WeightedLoss<T1, T2, TResult> : Loss<T1, T2, TResult>
    {
        public WeightedLoss(Tensor? weight = null, torch.nn.Reduction reduction = nn.Reduction.Mean) : base(reduction)
        {
            this.weight = weight;
        }

        public Tensor? weight { get; }
    }

    public abstract class WeightedLoss<T1, T2, T3, TResult> : Loss<T1, T2, T3, TResult>
    {
        public WeightedLoss(Tensor? weight = null, torch.nn.Reduction reduction = nn.Reduction.Mean) : base(reduction)
        {
            this.weight = weight;
        }

        public Tensor? weight { get; }
    }

    public abstract class WeightedLoss<T1, T2, T3, T4, TResult> : Loss<T1, T2, T3, T4, TResult>
    {
        public WeightedLoss(Tensor? weight = null, torch.nn.Reduction reduction = nn.Reduction.Mean) : base(reduction)
        {
            this.weight = weight;
        }

        public Tensor? weight { get; }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// This criterion combines log_softmax and nll_loss in a single function.
            /// </summary>
            /// <param name="weight">A manual rescaling weight if provided it’s repeated to match input tensor shape</param>
            /// <param name="ignore_index">Specifies a target value that is ignored and does not contribute to the input gradient.</param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static Modules.CrossEntropyLoss CrossEntropyLoss(Tensor? weight = null, long? ignore_index = null, Reduction reduction = Reduction.Mean)
            {
                return new Modules.CrossEntropyLoss(weight, ignore_index, reduction);
            }

            /// <summary>
            /// Measures the Binary Cross Entropy between the target and the output.
            /// </summary>
            /// <param name="weight">A manual rescaling weight if provided it’s repeated to match input tensor shape</param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static Modules.BCELoss BCELoss(Tensor? weight = null, Reduction reduction = Reduction.Mean)
            {
                return new Modules.BCELoss(weight, reduction);
            }

            /// <summary>
            /// Measures Binary Cross Entropy between target and output logits.
            /// </summary>
            /// <param name="weight">A manual rescaling weight if provided it’s repeated to match input tensor shape</param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <param name="pos_weights">A weight of positive examples. Must be a vector with length equal to the number of classes.</param>
            /// <returns></returns>
            public static Modules.BCEWithLogitsLoss BCEWithLogitsLoss(Tensor? weight = null, Reduction reduction = Reduction.Mean, Tensor? pos_weights = null)
            {
                return new Modules.BCEWithLogitsLoss(weight, reduction, pos_weights);
            }

            /// <summary>
            /// Measures the loss given two input tensor and a lable tensor with values 1 or -1.
            ///
            /// See: https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html#torch.nn.CosineEmbeddingLoss
            /// </summary>
            /// <param name="margin"> Should be a number from -1 to 1, 0 to 0.5 is suggested</param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static Modules.CosineEmbeddingLoss CosineEmbeddingLoss(double margin = 0.0, Reduction reduction = Reduction.Mean)
            {
                return new Modules.CosineEmbeddingLoss(margin, reduction);
            }


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
            public static Modules.CTCLoss CTCLoss(long blank = 0, bool zero_infinity = false, Reduction reduction = Reduction.Mean)
            {
                return new Modules.CTCLoss(blank, zero_infinity, reduction);
            }

            /// <summary>
            /// Measures the loss given an input tensor x and a labels tensor y (containing 1 or -1).
            ///
            /// See: https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html#torch.nn.HingeEmbeddingLoss
            /// </summary>
            /// <param name="margin"> Should be a number from -1 to 1, 0 to 0.5 is suggested</param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static Modules.HingeEmbeddingLoss HingeEmbeddingLoss(double margin = 1.0, Reduction reduction = Reduction.Mean)
            {
                return new Modules.HingeEmbeddingLoss(margin, reduction);
            }

            /// <summary>
            /// Creates a criterion that uses a squared term if the absolute element-wise error falls below delta and a delta-scaled L1 term otherwise.
            ///
            /// See: https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html#torch.nn.HuberLoss
            /// </summary>
            /// <param name="delta">Specifies the threshold at which to change between delta-scaled L1 and L2 loss. The value must be positive. Default: 1.0</param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static Modules.HuberLoss HuberLoss(double delta = 1.0, Reduction reduction = Reduction.Mean)
            {
                return new Modules.HuberLoss(delta, reduction);
            }

            /// <summary>
            /// Creates a criterion that measures the loss given inputs x1, x2, two 1D mini-batch or 0D Tensors, and a label 1D mini-batch or 0D Tensor y (containing 1 or -1).
            ///
            /// See: https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html#torch.nn.MarginRankingLoss
            /// </summary>
            /// <param name="margin">Has a default value of 0.</param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static Modules.MarginRankingLoss MarginRankingLoss(double margin = 0, Reduction reduction = Reduction.Mean)
            {
                return new Modules.MarginRankingLoss(margin, reduction);
            }

            /// <summary>
            /// Creates a criterion that optimizes a multi-label one-versus-all loss based on max-entropy, between input x and target y of size NxC.
            ///
            /// See: https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html#torch.nn.MultiLabelSoftMarginLoss
            /// </summary>
            /// <param name="weight">A manual rescaling weight given to each class. If given, it has to be a Tensor of size C. Otherwise, it is treated as if having all ones.</param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static MultiLabelSoftMarginLoss MultiLabelSoftMarginLoss(Tensor? weight = null, Reduction reduction = Reduction.Mean)
            {
                return new MultiLabelSoftMarginLoss(weight, reduction);
            }

            /// <summary>
            /// Creates a criterion that optimizes a multi-class classification hinge loss.
            ///
            /// See: https://pytorch.org/docs/stable/generated/torch.nn.MultiMarginLoss.html#torch.nn.MultiMarginLoss
            /// </summary>
            /// <param name="p">Has a default value of 1. 1 and 2 are the only supported values.</param>
            /// <param name="margin">Has a default value of 1</param>
            /// <param name="weight">A manual rescaling weight given to each class. If given, it has to be a Tensor of size C. Otherwise, it is treated as if having all ones.</param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            public static MultiMarginLoss MultiMarginLoss(int p = 1, double margin = 1.0, Tensor? weight = null, Reduction reduction = Reduction.Mean)
            {
                return new MultiMarginLoss(p, margin, weight, reduction);
            }

            /// <summary>
            /// Measures the element-wise mean squared error.
            /// </summary>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static MSELoss MSELoss(Reduction reduction = Reduction.Mean)
            {
                return new MSELoss(reduction);
            }

            /// <summary>
            /// Function that takes the mean element-wise absolute value difference.
            /// </summary>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static L1Loss L1Loss(Reduction reduction = Reduction.Mean)
            {
                return new L1Loss(reduction);
            }

            /// <summary>
            /// The negative log likelihood loss.
            /// </summary>
            /// <param name="weight">A manual rescaling weight if provided it’s repeated to match input tensor shape</param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static NLLLoss NLLLoss(Tensor? weight = null, Reduction reduction = Reduction.Mean)
            {
                return new NLLLoss(weight, reduction);
            }

            /// <summary>
            /// Poisson negative log likelihood loss.
            /// </summary>
            /// <param name="log_input"></param>
            /// <param name="full">Whether to compute full loss, i. e. to add the Stirling approximation term.</param>
            /// <param name="eps">Small value to avoid evaluation of log(0)</param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static PoissonNLLLoss PoissonNLLLoss(bool log_input = true, bool full = false, float eps = 1e-8f, Reduction reduction = Reduction.Mean)
            {
                return new PoissonNLLLoss(log_input, full, eps, reduction);
            }

            /// <summary>
            /// The Kullback-Leibler divergence Loss
            /// </summary>
            /// <param name="log_target">A flag indicating whether target is passed in the log space.</param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static KLDivLoss KLDivLoss(bool log_target = true, Reduction reduction = Reduction.Mean)
            {
                return new KLDivLoss(log_target, reduction);
            }

            /// <summary>
            /// Optimizes a two-class classification logistic loss between input tensor xx and target tensor yy (containing 1 or -1).
            /// </summary>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static SoftMarginLoss SoftMarginLoss(Reduction reduction = Reduction.Mean)
            {
                return new SoftMarginLoss(reduction);
            }

            /// <summary>
            /// Creates a criterion that measures the triplet loss given an input tensors x1, x2, x3 and a margin with a value greater than 0.
            /// This is used for measuring a relative similarity between samples.
            ///
            /// See: https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss
            /// </summary>
            /// <param name="margin">
            /// A nonnegative margin representing the minimum difference between the positive and negative distances required for the loss to be 0.
            /// Larger margins penalize cases where the negative examples are not distant enough from the anchors, relative to the positives.
            /// </param>
            /// <param name="p">The norm degree for pairwise distance. </param>
            /// <param name="eps"></param>
            /// <param name="swap">
            /// If true, and if the positive example is closer to the negative example than the anchor is, swaps the positive example and the anchor in the loss computation.
            /// The distance swap is described in detail in the paper Learning shallow convolutional feature descriptors with triplet losses by V. Balntas, E. Riba et al.
            /// </param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static TripletMarginLoss TripletMarginLoss(double margin = 1.0, long p = 2, double eps = 1e-06, bool swap = false, Reduction reduction = Reduction.Mean)
            {
                return new TripletMarginLoss(margin, p, eps, swap, reduction);
            }

            /// <summary>
            /// Creates a criterion that measures the triplet loss given input tensors a, p, and n (representing anchor, positive, and negative examples, respectively),
            /// and a nonnegative, real-valued function ("distance function") used to compute the relationship between the anchor and positive example ("positive distance")
            /// and the anchor and negative example ("negative distance").
            /// </summary>
            /// <param name="distance"> A nonnegative, real-valued function that quantifies the closeness of two tensors. If not specified, nn.PairwiseDistance will be used.</param>
            /// <param name="margin">
            /// A nonnegative margin representing the minimum difference between the positive and negative distances required for the loss to be 0.
            /// Larger margins penalize cases where the negative examples are not distant enough from the anchors, relative to the positives.
            /// </param>
            /// <param name="swap">
            /// If true, and if the positive example is closer to the negative example than the anchor is, swaps the positive example and the anchor in the loss computation.
            /// The distance swap is described in detail in the paper Learning shallow convolutional feature descriptors with triplet losses by V. Balntas, E. Riba et al.
            /// </param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static TripletMarginWithDistanceLoss TripletMarginWithDistanceLoss(Func<Tensor, Tensor, Tensor>? distance = null, double margin = 1.0, bool swap = false, Reduction reduction = Reduction.Mean)
            {
                return new TripletMarginWithDistanceLoss(distance, margin, swap, reduction);
            }

            /// <summary>
            /// Gaussian negative log likelihood loss.
            /// </summary>
            /// <param name="full">Include the constant term in the loss calculation</param>
            /// <param name="eps">Value used to clamp var (see note below), for stability.</param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static GaussianNLLLoss GaussianNLLLoss(bool full = false, float eps = 1e-8f, Reduction reduction = Reduction.Mean)
            {
                return new GaussianNLLLoss(full, eps, reduction);
            }

            /// <summary>
            /// Creates a criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss) between input x (a 2D mini-batch Tensor)
            /// and output y (which is a 2D Tensor of target class indices).
            /// </summary>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static MultiLabelMarginLoss MultiLabelMarginLoss(Reduction reduction = Reduction.Mean)
            {
                return new MultiLabelMarginLoss(reduction);
            }

            /// <summary>
            /// Function that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise.
            /// </summary>
            /// <param name="beta">Specifies the threshold at which to change between L1 and L2 loss. The value must be non-negative. Default: 1.0</param>
            /// <param name="reduction">Specifies the reduction to apply to the output</param>
            /// <returns></returns>
            public static SmoothL1Loss SmoothL1Loss(Reduction reduction = Reduction.Mean, double beta = 1.0)
            {
                return new SmoothL1Loss(reduction, beta);
            }


            /// <summary>
            /// Class maintaing the supported loss functions.
            /// </summary>
            public static partial class functional
            {
                /// <summary>
                /// Function that measures Binary Cross Entropy between target and input logits.
                /// </summary>
                /// <param name="input">Tensor of arbitrary shape as unnormalized scores (often referred to as logits).</param>
                /// <param name="target">Tensor of the same shape as input with values between 0 and 1</param>
                /// <param name="weight">A manual rescaling weight if provided it’s repeated to match input tensor shape</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <param name="pos_weights">A weight of positive examples. Must be a vector with length equal to the number of classes.</param>
                /// <returns></returns>
                public static Tensor binary_cross_entropy_with_logits(Tensor input, Tensor target, Tensor? weight = null, Reduction reduction = Reduction.Mean, Tensor? pos_weights = null)
                {
                    var res = THSNN_binary_cross_entropy_with_logits(input.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, (long)reduction, pos_weights?.Handle ?? IntPtr.Zero);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Function that measures the Binary Cross Entropy between the target and input probabilities.
                /// </summary>
                /// <param name="input">Tensor of arbitrary shape as probabilities.</param>
                /// <param name="target">Tensor of the same shape as input with values between 0 and 1</param>
                /// <param name="weight">A manual rescaling weight if provided it’s repeated to match input tensor shape</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor binary_cross_entropy(Tensor input, Tensor target, Tensor? weight = null, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_binary_cross_entropy(input.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Computes the cross entropy loss between input and target.
                /// </summary>
                /// <param name="input">Tensor of arbitrary shape as unnormalized scores (often referred to as logits).</param>
                /// <param name="target">Ground truth class indices or class probabilities; see Shape section below for supported shapes.</param>
                /// <param name="weight">A manual rescaling weight if provided it’s repeated to match input tensor shape</param>
                /// <param name="ignore_index">
                /// Specifies a target value that is ignored and does not contribute to the input gradient.
                /// Note that ignore_index is only applicable when the target contains class indices.
                /// </param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <param name="label_smoothing">A float in [0.0, 1.0].
                /// Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing.
                /// The targets become a mixture of the original ground truth and a uniform distribution.</param>
                /// <returns></returns>
                public static Tensor cross_entropy(Tensor input, Tensor target, Tensor? weight = null, long ignore_index = -100, Reduction reduction = Reduction.Mean, double label_smoothing = 0.0)
                {
                    var res = THSNN_cross_entropy(input.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, ignore_index, true, (long)reduction, label_smoothing);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Poisson negative log likelihood loss.
                /// </summary>
                /// <param name="input">Expectation of underlying Poisson distribution.</param>
                /// <param name="target">Random sample target.</param>
                /// <param name="log_input"></param>
                /// <param name="full">Whether to compute full loss, i.e. to add the Stirling approximation term.</param>
                /// <param name="eps">Small value to avoid evaluation of log(0)</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor poisson_nll_loss(Tensor input, Tensor target, bool log_input = true, bool full = false, float eps = 1e-8f, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_poisson_loss(input.Handle, target.Handle, log_input, full, eps, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                ///
                /// </summary>
                /// <param name="input1">(N,D) or (D), where N is the batch size and D is the embedding dimension.</param>
                /// <param name="input2">Same shape as input1</param>
                /// <param name="target">N or ()</param>
                /// <param name="margin">Should be a number from -1−1 to 11, 00 to 0.50.5 is suggested</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, double margin = 0.0, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_cosine_embedding_loss(input1.Handle, input2.Handle, target.Handle, margin, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Computes the Connectionist Temporal Classification loss.
                /// </summary>
                /// <param name="log_probs">The logarithmized probabilities of the outputs.</param>
                /// <param name="targets"></param>
                /// <param name="input_lengths">Lengths of the inputs.</param>
                /// <param name="target_lengths">Lengths of the targets.</param>
                /// <param name="blank">Blank label.</param>
                /// <param name="zero_infinity">Whether to zero infinite losses and the associated gradients.</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor ctc_loss(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, long blank = 0, bool zero_infinity = false, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_ctc_loss(log_probs.Handle, targets.Handle, input_lengths.Handle, target_lengths.Handle, blank, zero_infinity, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Measures the loss given an input tensor x and a labels tensor y (containing 1 or -1).
                /// </summary>
                /// <param name="input"></param>
                /// <param name="target"></param>
                /// <param name="margin"> Should be a number from -1 to 1, 0 to 0.5 is suggested</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor hinge_embedding_loss(Tensor input, Tensor target, double margin = 0.0, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_hinge_embedding_loss(input.Handle, target.Handle, margin, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Function that uses a squared term if the absolute element-wise error falls below delta and a delta-scaled L1 term otherwise.
                /// </summary>
                /// <param name="input"></param>
                /// <param name="target"></param>
                /// <param name="delta">Specifies the threshold at which to change between delta-scaled L1 and L2 loss. The value must be positive. Default: 1.0</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor huber_loss(Tensor input, Tensor target, double delta = 1.0, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_huber_loss(input.Handle, target.Handle, delta, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Creates a criterion that measures the loss given inputs x1, x2, two 1D mini-batch or 0D Tensors, and a label 1D mini-batch or 0D Tensor y (containing 1 or -1).
                /// </summary>
                /// <param name="input1"></param>
                /// <param name="input2"></param>
                /// <param name="target"></param>
                /// <param name="margin">Has a default value of 0.</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, double margin = 0.0, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_margin_ranking_loss(input1.Handle, input2.Handle, target.Handle, margin, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Creates a criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss) between input x (a 2D mini-batch Tensor)
                /// and output y (which is a 2D Tensor of target class indices).
                /// </summary>
                /// <param name="input"></param>
                /// <param name="target"></param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor multi_label_margin_loss(Tensor input, Tensor target, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_multilabel_margin_loss(input.Handle, target.Handle, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Creates a criterion that optimizes a multi-label one-versus-all loss based on max-entropy, between input x and target y of size NxC.
                /// </summary>
                /// <param name="input"></param>
                /// <param name="target"></param>
                /// <param name="weight">A manual rescaling weight if provided it’s repeated to match input tensor shape</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor multilabel_soft_margin_loss(Tensor input, Tensor target, Tensor? weight = null,Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_multilabel_soft_margin_loss(input.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Creates a criterion that optimizes a multi-class classification hinge loss.
                /// </summary>
                /// <param name="input"></param>
                /// <param name="target"></param>
                /// <param name="p">Has a default value of 1. 1 and 2 are the only supported values.</param>
                /// <param name="margin">Has a default value of 1</param>
                /// <param name="weight">A manual rescaling weight if provided it’s repeated to match input tensor shape</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor multi_margin_loss(Tensor input, Tensor target, int p = 1, double margin = 1.0, Tensor? weight = null, Reduction reduction = Reduction.Mean)
                {
                    IntPtr h = (weight is null) ? IntPtr.Zero : weight.Handle;
                    var res = THSNN_multi_margin_loss(input.Handle, target.Handle, p, margin, h, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                ///	Measures the element-wise mean squared error.
                /// </summary>
                /// <param name="input">Tensor of any shape.</param>
                /// <param name="target">Tensor of the same shape as 'input'</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor mse_loss(Tensor input, Tensor target, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_mse_loss(input.Handle, target.Handle, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Function that takes the mean element-wise absolute value difference.
                /// </summary>
                /// <param name="input">Tensor of any shape.</param>
                /// <param name="target">Tensor of the same shape as 'input'</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor l1_loss(Tensor input, Tensor target, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_l1_loss(input.Handle, target.Handle, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Computes the negative log likelihood loss.
                /// </summary>
                /// <param name="input"></param>
                /// <param name="target"></param>
                /// <param name="weight">A manual rescaling weight if provided it’s repeated to match input tensor shape</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor nll_loss(Tensor input, Tensor target, Tensor? weight = null, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_nll_loss(input.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Gaussian negative log likelihood loss.
                /// </summary>
                /// <param name="input"></param>
                /// <param name="target"></param>
                /// <param name="variance">Tensor of positive variance(s), one for each of the expectations in the input (heteroscedastic), or a single one (homoscedastic).</param>
                /// <param name="full">Include the constant term in the loss calculation. </param>
                /// <param name="eps">Value added to var, for stability</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor gaussian_nll_loss(Tensor input, Tensor target, Tensor variance, bool full = false, float eps = 1e-6f, Reduction reduction = Reduction.Mean)
                {
                    return new Modules.GaussianNLLLoss(full, eps, reduction).call(input, target, variance);
                }

                /// <summary>
                /// Computes the Kullback-Leibler divergence Loss
                /// </summary>
                /// <param name="input"></param>
                /// <param name="target"></param>
                /// <param name="log_target">A flag indicating whether target is passed in the log space.</param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor kl_div(Tensor input, Tensor target, bool log_target = true, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_kl_div_loss(input.Handle, target.Handle, (long)reduction, log_target);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Function that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise.
                /// </summary>
                /// <param name="input"></param>
                /// <param name="target"></param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <param name="beta">Specifies the threshold at which to change between L1 and L2 loss. The value must be non-negative.</param>
                /// <returns></returns>
                public static Tensor smooth_l1_loss(Tensor input, Tensor target, Reduction reduction = Reduction.Mean, double beta = 1.0)
                {
                    var res = THSNN_smooth_l1_loss(input.Handle, target.Handle, (long)reduction, beta);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                ///  Optimizes a two-class classification logistic loss between input tensor x and target tensor y (containing 1 or -1).
                /// </summary>
                /// <param name="input"></param>
                /// <param name="target"></param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor soft_margin_loss(Tensor input, Tensor target, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_soft_margin_loss(input.Handle, target.Handle, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Creates a criterion that measures the triplet loss given an input tensors x1, x2, x3 and a margin with a value greater than 0.
                /// This is used for measuring a relative similarity between samples.
                /// </summary>
                /// <param name="anchor"></param>
                /// <param name="positive"></param>
                /// <param name="negative"></param>
                /// <param name="margin">
                /// A nonnegative margin representing the minimum difference between the positive and negative distances required for the loss to be 0.
                /// Larger margins penalize cases where the negative examples are not distant enough from the anchors, relative to the positives.
                /// </param>
                /// <param name="p">The norm degree for pairwise distance. </param>
                /// <param name="eps"></param>
                /// <param name="swap">
                /// If true, and if the positive example is closer to the negative example than the anchor is, swaps the positive example and the anchor in the loss computation.
                /// The distance swap is described in detail in the paper Learning shallow convolutional feature descriptors with triplet losses by V. Balntas, E. Riba et al.
                /// </param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, double margin = 1.0, long p = 2, double eps = 1e-06, bool swap = false, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_triplet_margin_loss(anchor.Handle, positive.Handle, negative.Handle, margin, p, eps, swap, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Creates a criterion that measures the triplet loss given input tensors a, p, and n (representing anchor, positive, and negative examples, respectively),
                /// and a nonnegative, real-valued function ("distance function") used to compute the relationship between the anchor and positive example ("positive distance")
                /// and the anchor and negative example ("negative distance").
                /// </summary>
                /// <param name="anchor"></param>
                /// <param name="positive"></param>
                /// <param name="negative"></param>
                /// <param name="distance"> A nonnegative, real-valued function that quantifies the closeness of two tensors. If not specified, nn.PairwiseDistance will be used.</param>
                /// <param name="margin">
                /// A nonnegative margin representing the minimum difference between the positive and negative distances required for the loss to be 0.
                /// Larger margins penalize cases where the negative examples are not distant enough from the anchors, relative to the positives.
                /// </param>
                /// <param name="swap">
                /// If true, and if the positive example is closer to the negative example than the anchor is, swaps the positive example and the anchor in the loss computation.
                /// The distance swap is described in detail in the paper Learning shallow convolutional feature descriptors with triplet losses by V. Balntas, E. Riba et al.
                /// </param>
                /// <param name="reduction">Specifies the reduction to apply to the output</param>
                /// <returns></returns>
                public static Tensor triplet_margin_with_distance_loss(Tensor anchor, Tensor positive, Tensor negative, Func<Tensor, Tensor, Tensor>? distance = null, double margin = 1.0, bool swap = false, Reduction reduction = Reduction.Mean)
                {
                    DistanceFunctionNative? func = null;

                    if (distance != null) {
                        func = (IntPtr x, IntPtr y) => {
                            var x1 = new Tensor(x);
                            var y1 = new Tensor(y);
                            var res = distance(x1, y1);

                            GC.SuppressFinalize(x1);
                            GC.SuppressFinalize(y1);
                            GC.SuppressFinalize(res);

                            return res.Handle;
                        };
                    }
                    var res = THSNN_triplet_margin_with_distance_loss(anchor.Handle, positive.Handle, negative.Handle, func, margin, swap, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            public enum Reduction : long
            {
                None = 0,
                Mean = 1,
                Sum = 2
            }
        }
    }

    namespace Modules
    {
        public sealed class CrossEntropyLoss : WeightedLoss<Tensor, Tensor, Tensor>
        {
            public CrossEntropyLoss(Tensor? weight = null, long? ignore_index = null, Reduction reduction = Reduction.Mean, double label_smoothing = 0.0) : base(weight, reduction)
            {
                this.ignore_index = ignore_index;
                this.label_smoothing = label_smoothing;
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var ii = ignore_index.HasValue ? ignore_index.Value : -100;
                var res = THSNN_cross_entropy(input.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, ii, ignore_index.HasValue, (long)reduction, label_smoothing);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public long? ignore_index { get; }
            public double label_smoothing { get; }
        }

        public sealed class BCELoss : WeightedLoss<Tensor, Tensor, Tensor>
        {
            public BCELoss(Tensor? weight = null, Reduction reduction = Reduction.Mean) : base(weight, reduction)
            {
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var res = THSNN_binary_cross_entropy(input.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        public sealed class BCEWithLogitsLoss : WeightedLoss<Tensor, Tensor, Tensor>
        {
            public BCEWithLogitsLoss(Tensor? weight = null, Reduction reduction = Reduction.Mean, Tensor? pos_weights = null) : base(weight, reduction)
            {
                this.pos_weights = pos_weights;
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var res = THSNN_binary_cross_entropy_with_logits(input.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, (long)reduction, pos_weights?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor? pos_weights { get; }
        }

        public sealed class CosineEmbeddingLoss : Loss<Tensor, Tensor, Tensor, Tensor>
        {
            public CosineEmbeddingLoss(double margin = 0.0, Reduction reduction = Reduction.Mean) : base(reduction)
            {
                this.margin = margin;
            }

            public override Tensor forward(Tensor input1, Tensor input2, Tensor target)
            {
                var res = THSNN_cosine_embedding_loss(input1.Handle, input2.Handle, target.Handle, margin, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public double margin { get; }
        }

        public sealed class CTCLoss : Loss<Tensor, Tensor, Tensor, Tensor, Tensor>
        {
            public CTCLoss(long blank = 0, bool zero_infinity = false, Reduction reduction = Reduction.Mean) : base(reduction)
            {
                this.blank = blank;
                this.zero_infinity = zero_infinity;
            }

            public override Tensor forward(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths)
            {
                var res = THSNN_ctc_loss(log_probs.Handle, targets.Handle, input_lengths.Handle, target_lengths.Handle, blank, zero_infinity, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public long blank { get; }
            public bool zero_infinity { get; }
        }

        public sealed class HingeEmbeddingLoss : Loss<Tensor, Tensor, Tensor>
        {
            public HingeEmbeddingLoss(double margin = 0.0, Reduction reduction = Reduction.Mean) : base(reduction)
            {
                this.margin = margin;
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var res = THSNN_hinge_embedding_loss(input.Handle, target.Handle, margin, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public double margin { get; }
        }

        public sealed class HuberLoss : Loss<Tensor, Tensor, Tensor>
        {
            public HuberLoss(double delta = 1.0, Reduction reduction = Reduction.Mean) : base(reduction)
            {
                this.delta = delta;
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var res = THSNN_huber_loss(input.Handle, target.Handle, delta, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public double delta { get; }
        }

        public sealed class MarginRankingLoss : Loss<Tensor, Tensor, Tensor, Tensor>
        {
            public MarginRankingLoss(double margin = 0.0, Reduction reduction = Reduction.Mean) : base(reduction)
            {
                this.margin = margin;
            }

            public override Tensor forward(Tensor input1, Tensor input2, Tensor target)
            {
                var res = THSNN_margin_ranking_loss(input1.Handle, input2.Handle, target.Handle, margin, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public double margin { get; }
        }

        public sealed class MultiLabelMarginLoss : Loss<Tensor, Tensor, Tensor>
        {
            public MultiLabelMarginLoss(Reduction reduction = Reduction.Mean) : base(reduction)
            {
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var res = THSNN_multilabel_margin_loss(input.Handle, target.Handle, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        public sealed class MultiLabelSoftMarginLoss : WeightedLoss<Tensor, Tensor, Tensor>
        {
            public MultiLabelSoftMarginLoss(Tensor? weight = null, Reduction reduction = Reduction.Mean) : base(weight, reduction)
            {
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var res = THSNN_multilabel_soft_margin_loss(input.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        public sealed class MultiMarginLoss : WeightedLoss<Tensor, Tensor, Tensor>
        {
            public MultiMarginLoss(int p = 1, double margin = 1.0, Tensor? weight = null, Reduction reduction = Reduction.Mean) : base(weight, reduction)
            {
                this.margin = margin;
                this.p = p;
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                IntPtr h = (weight is null) ? IntPtr.Zero : weight.Handle;

                var res = THSNN_multi_margin_loss(input.Handle, target.Handle, p, margin, h, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public double margin { get; }
            public int p { get; }
        }

        public sealed class MSELoss : Loss<Tensor, Tensor, Tensor>
        {
            public MSELoss(Reduction reduction = Reduction.Mean) : base(reduction)
            {
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var res = THSNN_mse_loss(input.Handle, target.Handle, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        public sealed class L1Loss : Loss<Tensor, Tensor, Tensor>
        {
            public L1Loss(Reduction reduction = Reduction.Mean) : base(reduction)
            {
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var res = THSNN_l1_loss(input.Handle, target.Handle, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        public sealed class NLLLoss : WeightedLoss<Tensor, Tensor, Tensor>
        {
            public NLLLoss(Tensor? weight = null, Reduction reduction = Reduction.Mean) : base(weight, reduction)
            {
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var res = THSNN_nll_loss(input.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        public sealed class PoissonNLLLoss : Loss<Tensor, Tensor, Tensor>
        {
            public PoissonNLLLoss(bool log_input = true, bool full = false, float eps = 1e-8f, Reduction reduction = Reduction.Mean) : base(reduction)
            {
                this.full = full;
                this.log_input = log_input;
                this.eps = eps;
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var res = THSNN_poisson_loss(input.Handle, target.Handle, log_input, full, eps, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public bool log_input { get; }
            public bool full { get; }
            public float eps { get; }

        }

        public sealed class GaussianNLLLoss : Loss<Tensor, Tensor, Tensor, Tensor>
        {
            public GaussianNLLLoss(bool full = false, float eps = 1e-8f, Reduction reduction = Reduction.Mean) : base(reduction)
            {
                this.full = full;
                this.eps = eps;
            }

            public override Tensor forward(Tensor input, Tensor target, Tensor variance)
            {
                input = input.view(input.shape[0], -1);
                target = target.view(target.shape[0], -1);
                if (target.shape == input.shape) throw new ArgumentException("input and target must have the same shape");

                variance = variance.view(target.shape[0], -1);
                if (variance.shape[1] != input.shape[1] && variance.shape[1] != 1) throw new ArgumentException("variance has the wrong shape");

                if ((variance < 0).any().cpu().item<bool>()) throw new ArgumentException("variance has negative entry/entries");

                using (var _ = torch.no_grad())
                    variance = variance.clamp_min(eps);

                var loss = 0.5 * (variance.log() + (input - target).square() / variance).view(input.shape[0], -1).sum(dim: stackalloc long[] { 1 });

                if (full) {
                    loss = loss + 0.5 * input.shape[1] * MathF.Log(2 * MathF.PI);
                }

                return (reduction == Reduction.Mean) ? loss.mean() : (reduction == Reduction.Sum) ? loss.sum() : loss;
            }

            public bool full { get; }
            public float eps { get; }

        }

        public sealed class KLDivLoss : Loss<Tensor, Tensor, Tensor>
        {
            public KLDivLoss(bool log_target = true, Reduction reduction = Reduction.Mean) : base(reduction)
            {
                this.log_target = log_target;
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var res = THSNN_kl_div_loss(input.Handle, target.Handle, (long)reduction, log_target);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public bool log_target { get; }
        }

        public sealed class SmoothL1Loss : Loss<Tensor, Tensor, Tensor>
        {
            public SmoothL1Loss(Reduction reduction = Reduction.Mean, double beta = 1.0) : base(reduction)
            {
                this.beta = beta;
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var res = THSNN_smooth_l1_loss(input.Handle, target.Handle, (long)reduction, beta);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public double beta { get; }
        }

        public sealed class SoftMarginLoss : Loss<Tensor, Tensor, Tensor>
        {
            public SoftMarginLoss(Reduction reduction = Reduction.Mean) : base(reduction)
            {
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var res = THSNN_soft_margin_loss(input.Handle, target.Handle, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        public sealed class TripletMarginLoss : Loss<Tensor, Tensor, Tensor, Tensor>
        {
            public TripletMarginLoss(double margin = 1.0, long p = 2, double eps = 1e-06, bool swap = false, Reduction reduction = Reduction.Mean) : base(reduction)
            {
                this.margin = margin;
                this.p = p;
                this.eps = eps;
                this.swap = swap;
            }

            public override Tensor forward(Tensor anchor, Tensor positive, Tensor negative)
            {
                var res = THSNN_triplet_margin_loss(anchor.Handle, positive.Handle, negative.Handle, margin, p, eps, swap, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public double margin { get; }
            public long p { get; }
            double eps { get; }
            bool swap { get; }
        }

        public sealed class TripletMarginWithDistanceLoss : Loss<Tensor, Tensor, Tensor, Tensor>
        {
            public TripletMarginWithDistanceLoss(Func<Tensor, Tensor, Tensor>? distance = null, double margin = 1.0, bool swap = false, Reduction reduction = Reduction.Mean) : base(reduction)
            {
                if (distance != null) {
                    this.distance = (IntPtr x, IntPtr y) => {
                        var x1 = new Tensor(x);
                        var y1 = new Tensor(y);
                        var res = distance(x1, y1);

                        GC.SuppressFinalize(x1);
                        GC.SuppressFinalize(y1);
                        GC.SuppressFinalize(res);

                        return res.Handle;
                    };
                }

                this.margin = margin;
                this.swap = swap;
            }

            public override Tensor forward(Tensor anchor, Tensor positive, Tensor negative)
            {
                var res = THSNN_triplet_margin_with_distance_loss(anchor.Handle, positive.Handle, negative.Handle, distance, margin, swap, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            DistanceFunctionNative? distance { get; }
            public double margin { get; }
            bool swap { get; }
        }
    }
}
