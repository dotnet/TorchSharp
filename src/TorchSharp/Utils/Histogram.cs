// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;

namespace TorchSharp.Utils
{
    // https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/histograms.py
    internal static class Histogram
    {
        public static (Tensor hist, Tensor bin_edges) histogram(Tensor input, HistogramBinSelector bins, (double min, double max)? range, bool density = false)
        {
            input = RavelAndCheckWeights(input.cpu());
            (Tensor bin_edges, (double, double, int) uniform_bins) = GetBinEdges(input, bins, range);
            ScalarType ntype = ScalarType.Int32;
            int block = 65536;


            (double first_edge, double last_edge, int n_equal_bins) = uniform_bins;
            Tensor n = zeros(n_equal_bins, ntype);
            Tensor norm = n_equal_bins / subtract(last_edge, first_edge);

            for (int i = 0; i < input.shape[0]; i += block) {
                Tensor tmp_a = input[TensorIndex.Slice(i, i + block)];
                Tensor keep = (tmp_a >= first_edge);
                keep &= (tmp_a <= last_edge);
                if (keep.sum().item<long>() != tmp_a.numel())
                    tmp_a = tmp_a.masked_select(keep);

                tmp_a = tmp_a.to_type(bin_edges.dtype);
                Tensor f_indices = subtract(tmp_a, first_edge) * norm;
                Tensor indices = f_indices.to_type(ScalarType.Int64);
                indices[indices == n_equal_bins] -= 1;

                Tensor decrement = tmp_a < bin_edges[indices];
                indices[decrement] -= 1;
                Tensor increment = ((tmp_a >= bin_edges[indices + 1]) & (indices != n_equal_bins - 1));
                indices[increment] += 1;
                n += bincount(indices, minlength: n_equal_bins).to_type(ntype);
            }

            if (density) {
                Tensor db = diff(bin_edges).to_type(ScalarType.Float32);
                return (n / db / n.sum(), bin_edges);
            }
            return (n, bin_edges);
        }

        /// <summary>
        /// Computes the bins used internally by `histogram`.
        /// </summary>
        /// <param name="a"> Ravelled data array </param>
        /// <param name="bins"> Forwarded arguments from `histogram`. </param>
        /// <param name="range"> Ravelled weights array, or None </param>
        /// <returns></returns>
        private static (Tensor, (double, double, int)) GetBinEdges(Tensor a, HistogramBinSelector bins, (double min, double max)? range)
        {
            (double first_edge, double last_edge) = GetOuterEdges(a, range);
            if (range is not null) {
                Tensor keep = (a >= first_edge);
                keep &= (a <= last_edge);
                if (keep.sum().item<long>() != a.numel())
                    a = a.masked_select(keep);
            }

            int n_equal_bins;
            if (a.numel() == 0)
                n_equal_bins = 1;
            else {
                if (a.dtype != ScalarType.Float64)
                    a = a.to_type(ScalarType.Float64);
                Tensor width = histBinSelectors[bins](a, range);
                if ((width > 0).item<bool>())
                    n_equal_bins = ceil(subtract(last_edge, first_edge) / width).to_type(ScalarType.Int32).item<int>();
                else
                    n_equal_bins = 1;
            }

            Tensor bin_edges = linspace(first_edge, last_edge, n_equal_bins + 1, ScalarType.Float64, requires_grad: true);
            return (bin_edges, (first_edge, last_edge, n_equal_bins));
        }

        /// <summary>
        /// Determine the outer bin edges to use, from either the data or the range argument
        /// </summary>
        /// <param name="a"></param>
        /// <param name="range"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        private static (double, double) GetOuterEdges(Tensor a, (double min, double max)? range)
        {
            double first_edge, last_edge;
            if (range is not null) {
                (first_edge, last_edge) = range.Value;
                if (first_edge > last_edge)
                    throw new ArgumentException("max must be larger than min in range parameter.");
                if (double.IsInfinity(first_edge) || double.IsNaN(first_edge) || double.IsInfinity(last_edge) || double.IsNaN(last_edge))
                    throw new ArgumentException($"supplied range of [{first_edge}, {last_edge}] is not finite");
            } else if (a.numel() == 0) {
                (first_edge, last_edge) = (0, 1);
            } else {
                (first_edge, last_edge) = (a.min().to_type(ScalarType.Float64).item<double>(), a.max().to_type(ScalarType.Float64).item<double>());
                if (double.IsInfinity(first_edge) || double.IsNaN(first_edge) || double.IsInfinity(last_edge) || double.IsNaN(last_edge))
                    throw new ArgumentException($"autodetected range of [{first_edge}, {last_edge}] is not finite");
            }

            if (first_edge == last_edge)
                (first_edge, last_edge) = (first_edge - 0.5, last_edge + 0.5);
            return (first_edge, last_edge);
        }

        /// <summary>
        /// Check a and weights have matching shapes, and ravel both
        ///
        /// https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/histograms.py#L283
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private static Tensor RavelAndCheckWeights(Tensor input)
        {
            if (input.dtype == ScalarType.Bool)
                input = input.to_type(ScalarType.Int8);
            input = input.ravel();

            return input;
        }

        #region hist_bin
        private static Dictionary<HistogramBinSelector, Func<Tensor, (double min, double max)?, Tensor>> histBinSelectors
            = new Dictionary<HistogramBinSelector, Func<Tensor, (double min, double max)?, Tensor>>()
            {
                { HistogramBinSelector.Stone, HistBinStone },
                { HistogramBinSelector.Doane, HistBinDoane },
                { HistogramBinSelector.Rice, HistBinRice },
                { HistogramBinSelector.Scott, HistBinScott },
                { HistogramBinSelector.Sqrt, HistBinSqrt },
                { HistogramBinSelector.Sturges, HistBinSturges },
            };

        /// <summary>
        /// Square root histogram bin estimator.
        ///
        /// Bin width is inversely proportional to the data size. Used by many
        /// programs for its simplicity.
        ///
        /// https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/histograms.py#L32
        /// </summary>
        /// <param name="x"> Input data that is to be histogrammed, trimmed to range. May not be empty. </param>
        /// <param name="_"></param>
        /// <returns> An estimate of the optimal bin width for the given data. </returns>
        private static Tensor HistBinSqrt(Tensor x, (double min, double max)? _)
            => Ptp(x) / sqrt(x.numel());

        /// <summary>
        /// Sturges histogram bin estimator.
        ///
        /// A very simplistic estimator based on the assumption of normality of
        /// the data.This estimator has poor performance for non-normal data,
        /// which becomes especially obvious for large data sets.The estimate
        /// depends only on size of the data.
        ///
        /// https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/histograms.py#L53
        /// </summary>
        /// <param name="x"> Input data that is to be histogrammed, trimmed to range. May not be empty. </param>
        /// <param name="_"></param>
        /// <returns> An estimate of the optimal bin width for the given data. </returns>
        private static Tensor HistBinSturges(Tensor x, (double min, double max)? _)
            => Ptp(x) / (log2(x.numel()) + 1);

        /// <summary>
        /// Rice histogram bin estimator.
        ///
        /// Another simple estimator with no normality assumption. It has better
        /// performance for large data than Sturges, but tends to overestimate
        /// the number of bins. The number of bins is proportional to the cube
        /// root of data size (asymptotically optimal). The estimate depends
        /// only on size of the data.
        ///
        /// https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/histograms.py#L76
        /// </summary>
        /// <param name="x"> Input data that is to be histogrammed, trimmed to range. May not be empty. </param>
        /// <param name="_"></param>
        /// <returns> An estimate of the optimal bin width for the given data. </returns>
        private static Tensor HistBinRice(Tensor x, (double min, double max)? _)
            => Ptp(x) / (2 * pow(x.numel(), 1.0 / 3.0));

        /// <summary>
        /// Scott histogram bin estimator.
        ///
        /// The binwidth is proportional to the standard deviation of the data
        /// and inversely proportional to the cube root of data size
        /// (asymptotically optimal).
        ///
        /// https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/histograms.py#L100
        /// </summary>
        /// <param name="x"> Input data that is to be histogrammed, trimmed to range. May not be empty. </param>
        /// <param name="_"></param>
        /// <returns> An estimate of the optimal bin width for the given data. </returns>
        private static Tensor HistBinScott(Tensor x, (double min, double max)? _)
            => Math.Pow(24.0 * Math.Pow(Math.PI, 0.5) / x.numel(), 1.0 / 3.0) * std(x, false);

        /// <summary>
        /// Histogram bin estimator based on minimizing the estimated integrated squared error (ISE).
        ///
        /// The number of bins is chosen by minimizing the estimated ISE against the unknown true distribution.
        /// The ISE is estimated using cross-validation and can be regarded as a generalization of Scott's rule.
        /// https://en.wikipedia.org/wiki/Histogram#Scott.27s_normal_reference_rule
        /// 
        /// This paper by Stone appears to be the origination of this rule.
        /// http://digitalassets.lib.berkeley.edu/sdtr/ucb/text/34.pdf
        ///
        /// https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/histograms.py#L122
        /// </summary>
        /// <param name="x"> Input data that is to be histogrammed, trimmed to range. May not be empty. </param>
        /// <param name="range"> The lower and upper range of the bins. </param>
        /// <returns> An estimate of the optimal bin width for the given data. </returns>
        private static Tensor HistBinStone(Tensor x, (double min, double max)? range)
        {
            long n = x.numel();
            Tensor ptp_x = Ptp(x);
            if (n <= 1 || (ptp_x == 0).item<bool>())
                return 0;

            double Jhat(int nbins)
            {
                Tensor hh = ptp_x / nbins;
                Tensor pk = torch.histogram(x, bins: nbins, range: range).hist / n;
                return ((2 - (n + 1) * pk.dot(pk)) / hh).to_type(ScalarType.Float64).item<double>();
            }

            int nbinsUpperBound = Math.Max(100, Convert.ToInt32(Math.Sqrt(n)));
            int nbins = 0;
            double jhatTemp = double.PositiveInfinity;
            foreach (int item in Enumerable.Range(1, nbinsUpperBound + 1)) {
                double jhat = Jhat(item);
                if (jhat < jhatTemp) {
                    jhatTemp = jhat;
                    nbins = item;
                }
            }
            return ptp_x / nbins;
        }

        /// <summary>
        /// Doane's histogram bin estimator.
        ///
        /// Improved version of Sturges' formula which works better for
        /// non-normal data. See
        /// stats.stackexchange.com/questions/55134/doanes-formula-for-histogram-binning
        ///
        /// https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/histograms.py#L164
        /// </summary>
        /// <param name="x"> Input data that is to be histogrammed, trimmed to range. May not be empty. </param>
        /// <param name="_"></param>
        /// <returns> An estimate of the optimal bin width for the given data. </returns>
        private static Tensor HistBinDoane(Tensor x, (double min, double max)? _)
        {
            long size = x.numel();
            if (size > 2) {
                Tensor sg1 = sqrt(6.0 * (size - 2) / ((size + 1.0) * (size + 3)));
                Tensor sigma = x.std();
                if ((sigma > 0.0).item<bool>()) {
                    Tensor temp = x - x.mean();
                    temp = true_divide(temp, sigma);
                    temp = float_power(temp, 3);
                    Tensor g1 = temp.mean();
                    return Ptp(x) / (1.0 + log2(size) + log2(1.0 + absolute(g1) / sg1));
                }
            }
            return 0.0;
        }
        #endregion

        /// <summary>
        /// This implementation avoids the problem of signed integer arrays having a
        /// peak-to-peak value that cannot be represented with the array's data type.
        /// This function returns an value for signed integer arrays.
        ///
        /// https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/histograms.py#L22
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private static Tensor Ptp(Tensor input)
            => subtract(input.max(), input.min());
    }
}
