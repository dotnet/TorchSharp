// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Diagnostics.Contracts;
using TorchSharp.PInvoke;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    // https://pytorch.org/docs/stable/torch#random-sampling
    public static partial class torch
    {
        // https://pytorch.org/docs/stable/generated/torch.seed
        /// <summary>
        /// Sets the seed for generating random numbers to a non-deterministic random number. Returns a 64 bit number used to seed the RNG.
        /// </summary>
        public static long seed() => torch.random.seed();

        // https://pytorch.org/docs/stable/generated/torch.manual_seed
        /// <summary>
        /// Sets the seed for generating random numbers. Returns a torch.Generator object.
        /// </summary>
        /// <param name="seed">The desired seed.</param>
        public static Generator manual_seed(long seed) => torch.random.manual_seed(seed);

        // https://pytorch.org/docs/stable/generated/torch.initial_seed
        /// <summary>
        /// Returns the initial seed for generating random numbers.
        /// </summary>
        public static long initial_seed() => torch.random.initial_seed();

        // https://pytorch.org/docs/stable/generated/torch.get_rng_state
        /// <summary>
        /// Returns the random number generator state as a torch.ByteTensor.
        /// </summary>
        public static Tensor get_rng_state() => torch.random.get_rng_state();

        // https://pytorch.org/docs/stable/generated/torch.set_rng_state
        /// <summary>
        /// Sets the random number generator state.
        /// </summary>
        /// <param name="new_state">The desired state</param>
        public static void set_rng_state(Tensor new_state) => torch.random.set_rng_state(new_state);

        // https://pytorch.org/docs/stable/generated/torch.bernoulli
        /// <summary>
        /// Draws binary random numbers (0 or 1) from a Bernoulli distribution.
        /// </summary>
        /// <param name="input">The input tensor of probability values for the Bernoulli distribution</param>
        /// <param name="generator">Optional random number generator</param>
        /// <returns></returns>
        public static Tensor bernoulli(Tensor input, Generator? generator = null) => input.bernoulli(generator);

        // https://pytorch.org/docs/stable/generated/torch.multinomial
        /// <summary>
        /// Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.
        /// </summary>
        /// <param name="input">A probabilities tensor</param>
        /// <param name="num_samples">Number of samples to draw</param>
        /// <param name="replacement">Whether to draw with replacement or not</param>
        /// <param name="generator">Optional random number generator</param>
        public static Tensor multinomial(Tensor input, long num_samples, bool replacement = false, Generator? generator = null) => input.multinomial(num_samples, replacement, generator);

        // https://pytorch.org/docs/stable/generated/torch.normal
        /// <summary>
        /// Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.
        /// </summary>
        /// <param name="mean">The tensor of per-element means</param>
        /// <param name="std">The tensor of per-element standard deviations</param>
        /// <param name="generator">An optional random number generator</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor normal(Tensor mean, Tensor std, Generator? generator = null)
        {
            if (std.device_type != mean.device_type || (std.device_type == DeviceType.CUDA && std.device_index != mean.device_index))
                throw new ArgumentException("The 'means' and 'stddev' tensors must be located on the same device.");
            return randn(mean.shape, null, std.device, false, generator, names: null) * std + mean;
        }

        // https://pytorch.org/docs/stable/generated/torch.poisson
        /// <summary>
        /// Returns a tensor of the same size as input with each element sampled from a Poisson distribution with rate parameter given by the corresponding element in input
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <param name="generator">Optional random number generator</param>
        /// <returns></returns>
        public static Tensor poisson(Tensor input, Generator? generator = null) => input.poisson(generator);

        // https://pytorch.org/docs/stable/generated/torch.rand
        // TODO: implement layout parameter
        static Tensor rand(
            long[] size,
            ScalarType? dtype = null,
            layout layout = layout.strided,
            Device? device = null,
            bool requires_grad = false)
            => rand(size, dtype, device, requires_grad);

        // https://pytorch.org/docs/stable/generated/torch.rand_like
        // TODO: implement layout parameter
        // TODO: implement memory_format parameter
        static Tensor rand_like(
            Tensor input,
            ScalarType? dtype=null,
            layout layout = layout.strided,
            Device? device=null,
            bool requires_grad=false,
            memory_format memory_format = memory_format.preserve_format)
        => rand_like(input, dtype, device, requires_grad);

        // https://pytorch.org/docs/stable/generated/torch.randint
        // TODO: implement layout parameter
        static Tensor randint(long low, long high, Size size,
            Generator? generator = null,
            ScalarType? dtype = null,
            layout layout = layout.strided,
            Device? device = null,
            bool requires_grad = false)
            => randint(low, high, size, dtype, device, requires_grad, generator);

        // TODO: implement layout parameter
        static Tensor randint(long high, Size size,
            Generator? generator = null,
            ScalarType? dtype = null,
            layout layout = layout.strided,
            Device? device = null,
            bool requires_grad = false)
            => randint(high, size, dtype, device, requires_grad, generator);

        // https://pytorch.org/docs/stable/generated/torch.randint_like
        // TODO: implement layout parameter
        // TODO: implement memory_format parameter
        static Tensor randint_like(
            Tensor input,
            long low,
            long high,
            ScalarType? dtype = null,
            layout layout = layout.strided,
            Device? device = null,
            bool requires_grad = false,
            memory_format memory_format = memory_format.preserve_format)
            => randint_like(input, low, high, dtype, device, requires_grad);

        // https://pytorch.org/docs/stable/generated/torch.randn
        // TODO: implement layout parameter
        static Tensor randn(
            long[] size,
            Generator? generator = null,
            ScalarType? dtype = null,
            layout layout = layout.strided,
            Device? device = null,
            bool requires_grad = false)
            => randn(size, dtype, device, requires_grad, generator);

        // https://pytorch.org/docs/stable/generated/torch.randn_like
        /// <summary>
        /// Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval [0,1) .
        /// </summary>
        // TODO: implement layout parameter
        // TODO: implement memory_format parameter
        static Tensor randn_like(
            Tensor input,
            ScalarType? dtype = null,
            layout layout = layout.strided,
            Device? device = null,
            bool requires_grad = false,
            memory_format memory_format = memory_format.preserve_format)
            => input.rand_like(dtype, device, requires_grad);

        // https://pytorch.org/docs/stable/generated/torch.randn_like
        /// <summary>
        /// Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval [0,1) .
        /// </summary>
        public static Tensor rand_like(Tensor input, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
            => input.rand_like(dtype, device, requires_grad);

        // https://pytorch.org/docs/stable/generated/torch.randperm
        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        public static Tensor randperm(long n, ScalarType? dtype = null, Device? device = null,
            bool requires_grad = false, Generator? generator = null)
            => randperm(n, generator, dtype, layout.strided, device, false);

        /// <summary>
        /// Mutates the existing tensor to be a 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        public static Tensor randperm(long n,
            Tensor @out,
            Generator? generator = null)
        {
            var genHandle = generator?.Handle ?? IntPtr.Zero;
            var res = NativeMethods.THSTensor_randperm_out(genHandle, n, @out.Handle);
            if (res == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(res);
        }

        // https://pytorch.org/docs/stable/generated/torch.randperm
        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        // TODO: implement layout parameter
        // TODO: implement pin_memory parameter
        static Tensor randperm(
            long n,
            Generator? generator = null,
            ScalarType? dtype = ScalarType.Int64,
            layout layout = layout.strided,
            Device? device = null,
            bool requires_grad = false,
            bool pin_memory = false)
        {
            device = InitializeDevice(device);
            dtype ??= ScalarType.Int64;

            var genHandle = generator?.Handle ?? IntPtr.Zero;

            var handle = THSTensor_randperm(genHandle, n, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm(genHandle, n, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            }
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }
    }
}