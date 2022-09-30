// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

#nullable enable

namespace TorchSharp
{
    using Modules;

    public class Loss : nn.Module
    {
        public Loss(torch.nn.Reduction reduction = nn.Reduction.Mean) : base(nameof(Loss))
        {
            this.reduction = reduction;
        }

        public torch.nn.Reduction reduction { get; }
    }
    public class WeightedLoss : Loss
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
            public static Modules.HingeEmbeddingLoss HingeEmbeddingLoss(double margin = 0.0, Reduction reduction = Reduction.Mean)
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

            public static Modules.MarginRankingLoss MarginRankingLoss(double margin = 0, Reduction reduction = Reduction.Mean)
            {
                return new Modules.MarginRankingLoss(margin, reduction);
            }

            public static MultiLabelSoftMarginLoss MultiLabelSoftMarginLoss(Tensor? weight = null, Reduction reduction = Reduction.Mean)
            {
                return new MultiLabelSoftMarginLoss(weight, reduction);
            }

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

            public static TripletMarginLoss TripletMarginLoss(double margin = 1.0, long p = 2, double eps = 1e-06, bool swap = false, Reduction reduction = Reduction.Mean)
            {
                return new TripletMarginLoss(margin, p, eps, swap, reduction);
            }

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
                public static Tensor binary_cross_entropy_with_logits(Tensor input, Tensor target, Tensor? weight = null, Reduction reduction = Reduction.Mean, Tensor? posWeights = null)
                {
                    var res = THSNN_binary_cross_entropy_with_logits(input.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, (long)reduction, posWeights?.Handle ?? IntPtr.Zero);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                public static Tensor binary_cross_entropy(Tensor src, Tensor target, Tensor? weight = null, Reduction reduction = Reduction.Mean)
                {
                    var res = THSNN_binary_cross_entropy(src.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, (long)reduction);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
                public delegate IntPtr DistanceFunctionNative(IntPtr x, IntPtr y);

                #region External functions
                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_cross_entropy(IntPtr srct, IntPtr trgt, IntPtr wgt, long ignore_index, bool hasII, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_binary_cross_entropy(IntPtr srct, IntPtr trgt, IntPtr wgt, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_binary_cross_entropy_with_logits(IntPtr srct, IntPtr trgt, IntPtr wgt, long reduction, IntPtr posWeights);
                
                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_cosine_embedding_loss(IntPtr input1, IntPtr input2, IntPtr trgt, double margin, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_ctc_loss(IntPtr log_probs, IntPtr targets, IntPtr input_lengths, IntPtr target_lengths, long blank, bool zero_infinity, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_hinge_embedding_loss(IntPtr input, IntPtr trgt, double margin, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_huber_loss(IntPtr input, IntPtr trgt, double delta, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_margin_ranking_loss(IntPtr input1, IntPtr input2, IntPtr target, double margin, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_multilabel_margin_loss(IntPtr input, IntPtr target, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_multilabel_soft_margin_loss(IntPtr input, IntPtr target, IntPtr weight, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_multi_margin_loss(IntPtr input, IntPtr target, long p, double margin, IntPtr weight, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_mse_loss(IntPtr srct, IntPtr trgt, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_l1_loss(IntPtr srct, IntPtr trgt, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_nll_loss(IntPtr srct, IntPtr trgt, IntPtr wgt, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_poisson_loss(IntPtr srct, IntPtr trgt, bool logInput, bool full, float eps, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_kl_div_loss(IntPtr input, IntPtr target, long reduction, bool logTarget);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_smooth_l1_loss(IntPtr srct, IntPtr trgt, long reduction, double beta);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_soft_margin_loss(IntPtr srct, IntPtr trgt, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_triplet_margin_loss(IntPtr anchor, IntPtr positive, IntPtr negative, double margin, long p, double eps, bool swap, long reduction);

                [DllImport("LibTorchSharp")]
                internal static extern IntPtr THSNN_triplet_margin_with_distance_loss(IntPtr anchor, IntPtr positive, IntPtr negative, DistanceFunctionNative? distance_function, double margin, bool swap, long reduction);
                #endregion
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
        using static torch.nn.functional;

        public class CrossEntropyLoss : WeightedLoss
        {
            public CrossEntropyLoss(Tensor? weight = null, long? ignore_index = null, Reduction reduction = Reduction.Mean) : base(weight, reduction)
            {
                this.ignore_index = ignore_index;
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                var ii = ignore_index.HasValue ? ignore_index.Value : -100;
                var res = THSNN_cross_entropy(input.Handle, target.Handle, weight?.Handle ?? IntPtr.Zero, ii, ignore_index.HasValue, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public long? ignore_index { get; }
        }

        public class BCELoss : WeightedLoss
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

        public class BCEWithLogitsLoss : WeightedLoss
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

        public class CosineEmbeddingLoss : Loss
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

        public class CTCLoss : Loss
        {
            public CTCLoss(long blank = 0, bool zero_infinity = false, Reduction reduction = Reduction.Mean) : base(reduction)
            {
                this.blank = blank;
                this.zero_infinity = zero_infinity;
            }

            public Tensor forward(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths)
            {
                var res = THSNN_ctc_loss(log_probs.Handle, targets.Handle, input_lengths.Handle, target_lengths.Handle, blank, zero_infinity, (long)reduction);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public long blank { get; }
            public bool zero_infinity { get; }
        }

        public class HingeEmbeddingLoss : Loss
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

        public class HuberLoss : Loss
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

        public class MarginRankingLoss : Loss
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

        public class MultiLabelMarginLoss : Loss
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

        public class MultiLabelSoftMarginLoss : WeightedLoss
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

        public class MultiMarginLoss : WeightedLoss
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

        public class MSELoss : Loss
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

        public class L1Loss : Loss
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

        public class NLLLoss : WeightedLoss
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

        public class PoissonNLLLoss : Loss
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

        public class GaussianNLLLoss : Loss
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

        public class KLDivLoss : Loss
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

        public class SmoothL1Loss : Loss
        {
            public SmoothL1Loss(Reduction reduction = Reduction.Mean, double beta = 1.0) : base(reduction)
            {
                this.beta = beta;
            }

            public override Tensor forward(Tensor input, Tensor target)
            {
                // Currently, the 'beta' parameter is being ignored by the native layer, so we just pass the default.
                var res = THSNN_smooth_l1_loss(input.Handle, target.Handle, (long)reduction, 1.0);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public double beta { get; }
        }

        public class SoftMarginLoss : Loss
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

        public class TripletMarginLoss : Loss
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

        public class TripletMarginWithDistanceLoss : Loss
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
