// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Runtime.InteropServices;
using TorchSharp.PInvoke;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class export
        {
            /// <summary>
            /// Load a PyTorch ExportedProgram from a .pt2 file compiled with AOTInductor.
            /// </summary>
            /// <param name="filename">Path to the .pt2 file</param>
            /// <returns>ExportedProgram model for inference</returns>
            /// <remarks>
            /// IMPORTANT: The .pt2 file must be compiled with torch._inductor.aoti_compile_and_package() in Python.
            /// Models saved with torch.export.save() alone will NOT work - they require AOTInductor compilation.
            ///
            /// This implementation is INFERENCE-ONLY. Training, parameter updates, and device movement
            /// are not supported. The model is compiled for a specific device (CPU/CUDA) at compile time.
            ///
            /// Example Python code to create compatible .pt2 files:
            /// <code>
            /// import torch
            /// import torch._inductor
            ///
            /// # Export the model
            /// exported = torch.export.export(model, example_inputs)
            ///
            /// # Compile with AOTInductor (required for C++ loading)
            /// torch._inductor.aoti_compile_and_package(
            ///     exported,
            ///     package_path="model.pt2"
            /// )
            /// </code>
            /// </remarks>
            public static ExportedProgram load(string filename)
            {
                return new ExportedProgram(filename);
            }

            /// <summary>
            /// Load a PyTorch ExportedProgram with typed output.
            /// </summary>
            public static ExportedProgram<TResult> load<TResult>(string filename)
            {
                return new ExportedProgram<TResult>(filename);
            }
        }
    }

    /// <summary>
    /// Represents a PyTorch ExportedProgram loaded from an AOTInductor-compiled .pt2 file.
    /// This is an INFERENCE-ONLY implementation - training and parameter updates are not supported.
    /// </summary>
    /// <remarks>
    /// Unlike TorchScript models, ExportedProgram models are ahead-of-time (AOT) compiled for
    /// a specific device and are optimized for inference performance. They provide 30-40% better
    /// latency compared to TorchScript in many cases.
    ///
    /// Key limitations:
    /// - Inference only (no training, no gradients)
    /// - No parameter access or updates
    /// - No device movement (compiled for specific device)
    /// - No dynamic model structure changes
    ///
    /// Use torch.jit for models that require training or dynamic behavior.
    /// </remarks>
    public class ExportedProgram : IDisposable
    {
        private IntPtr handle;
        private bool _disposed = false;

        internal ExportedProgram(string filename)
        {
            handle = THSExport_load(filename);
            if (handle == IntPtr.Zero)
                torch.CheckForErrors();
        }

        /// <summary>
        /// Run inference on the model with the given input tensors.
        /// </summary>
        /// <param name="inputs">Input tensors for the model</param>
        /// <returns>Array of output tensors</returns>
        /// <remarks>
        /// The number and shapes of inputs must match what the model was exported with.
        /// All inputs must be on the same device that the model was compiled for.
        /// </remarks>
        public torch.Tensor[] run(params torch.Tensor[] inputs)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(ExportedProgram));

            // Convert managed tensors to IntPtr array
            IntPtr[] input_handles = new IntPtr[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                input_handles[i] = inputs[i].Handle;
            }

            // Call native run method
            THSExport_Module_run(handle, input_handles, inputs.Length, out IntPtr result_ptr, out int result_length);
            torch.CheckForErrors();

            // Marshal result array
            torch.Tensor[] results = new torch.Tensor[result_length];
            IntPtr[] result_handles = new IntPtr[result_length];
            Marshal.Copy(result_ptr, result_handles, 0, result_length);

            for (int i = 0; i < result_length; i++)
            {
                results[i] = new torch.Tensor(result_handles[i]);
            }

            // Free the native array (tensors are now owned by managed Tensor objects)
            Marshal.FreeHGlobal(result_ptr);

            return results;
        }

        /// <summary>
        /// Synonym for run() - executes forward pass.
        /// </summary>
        public torch.Tensor[] forward(params torch.Tensor[] inputs) => run(inputs);

        /// <summary>
        /// Synonym for run() - executes the model.
        /// </summary>
        public torch.Tensor[] call(params torch.Tensor[] inputs) => run(inputs);

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (handle != IntPtr.Zero)
                {
                    THSExport_Module_dispose(handle);
                    handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~ExportedProgram()
        {
            Dispose(false);
        }
    }

    /// <summary>
    /// Generic version of ExportedProgram with typed output.
    /// </summary>
    /// <typeparam name="TResult">The return type (Tensor, Tensor[], or tuple of Tensors)</typeparam>
    public class ExportedProgram<TResult> : ExportedProgram
    {
        internal ExportedProgram(string filename) : base(filename)
        {
        }

        /// <summary>
        /// Run inference with typed return value.
        /// </summary>
        public new TResult run(params torch.Tensor[] inputs)
        {
            var results = base.run(inputs);

            // Handle different return types
            if (typeof(TResult) == typeof(torch.Tensor))
            {
                if (results.Length != 1)
                    throw new InvalidOperationException($"Expected 1 output tensor, got {results.Length}");
                return (TResult)(object)results[0];
            }

            if (typeof(TResult) == typeof(torch.Tensor[]))
            {
                return (TResult)(object)results;
            }

            // Handle tuple types
            if (typeof(TResult).IsGenericType)
            {
                var genericType = typeof(TResult).GetGenericTypeDefinition();
                if (genericType == typeof(ValueTuple<,>))
                {
                    if (results.Length != 2)
                        throw new InvalidOperationException($"Expected 2 output tensors, got {results.Length}");
                    return (TResult)Activator.CreateInstance(typeof(TResult), results[0], results[1]);
                }
                if (genericType == typeof(ValueTuple<,,>))
                {
                    if (results.Length != 3)
                        throw new InvalidOperationException($"Expected 3 output tensors, got {results.Length}");
                    return (TResult)Activator.CreateInstance(typeof(TResult), results[0], results[1], results[2]);
                }
            }

            throw new NotSupportedException($"Return type {typeof(TResult)} is not supported");
        }

        public new TResult forward(params torch.Tensor[] inputs) => run(inputs);
        public new TResult call(params torch.Tensor[] inputs) => run(inputs);
    }
}
