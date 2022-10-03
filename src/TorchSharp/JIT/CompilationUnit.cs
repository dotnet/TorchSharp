// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using static TorchSharp.torch;
using System.Net;
using static TorchSharp.torch.nn;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class jit
        {
            /// <summary>
            /// Represents a TorchScript compilation unit, i.e. a Python script file.
            /// </summary>
            /// <example>
            /// var cu = torch.jit.compile(@"
            ///   def relu_script(a, b):
            ///     return torch.relu(a + b)
            /// ");
            ///
            /// var y = cu.invoke("relu_script", torch.randn(10));
            /// </example>
            /// <remarks>
            /// Currently, scripts are limited to defining functions. Classes will be ignored.
            /// </remarks>
            public class CompilationUnit : IDisposable
            {
                internal CompilationUnit(IntPtr handle) 
                {
                    this.handle = handle;
                }

                ~CompilationUnit() => Dispose(false);

                /// <summary>
                /// Releases the storage.
                /// </summary>
                public void Dispose()
                {
                    GC.SuppressFinalize(this);
                    Dispose(true);
                }

                /// <summary>
                /// Implements the .NET Dispose pattern.
                /// </summary>
                protected virtual void Dispose(bool disposing)
                {
                    if (disposing && handle != IntPtr.Zero) {
                        handle = IntPtr.Zero;
                    }
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSJIT_CompilationUnit_dispose(IntPtr handle);

                internal IntPtr handle;

                [DllImport("LibTorchSharp")]
                private static extern void THSJIT_CompilationUnit_Invoke(IntPtr module, string name, IntPtr tensors, int length, AllocatePinnedArray allocator, out sbyte typeCode);

                /// <summary>
                /// Invoke a function from the compilation unit.
                /// </summary>
                /// <param name="name">The name of the function.</param>
                /// <param name="objs">Function arguments.</param>
                public object invoke(string name, params object[] objs)
                {
                    if (String.IsNullOrEmpty(name)) throw new ArgumentNullException("method name");

                    if (!objs.All(o => typeof(Tensor).IsAssignableFrom(o.GetType()))) {
                        throw new NotImplementedException("ScriptModule.forward() taking non-tensors as input arguments");
                    }

                    IntPtr[] ptrArray = null;
                    sbyte typeCode = 0;

                    using (var parray = new PinnedArray<IntPtr>()) {

                        var tensors = objs.Select(o => (Tensor)o).ToArray();
                        var count = tensors.Length;
                        var tensorRefs = new IntPtr[count];
                        for (var i = 0; i < tensors.Length; i++) tensorRefs[i] = tensors[i].Handle;

                        THSJIT_CompilationUnit_Invoke(handle, name, parray.CreateArray(tensorRefs), count, parray.CreateArray, out typeCode);
                        torch.CheckForErrors();
                        ptrArray = parray.Array;
                    }


                    switch (typeCode) {
                    default:
                        // Nothing.
                        throw new NotImplementedException("ScriptModule.forward() returning something else than a tensor, a tuple of tensors, or list of tensors.");
                    case 1:
                        // Tensor
                        return new Tensor(ptrArray[0]);
                    case 2:
                        // Tuple
                        switch (ptrArray.Length) {
                        case 1:
                            return new Tensor(ptrArray[0]);
                        case 2:
                            return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
                        case 3:
                            return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]), new Tensor(ptrArray[2]));
                        case 4:
                            return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]), new Tensor(ptrArray[2]), new Tensor(ptrArray[3]));
                        case 5:
                            return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]), new Tensor(ptrArray[2]), new Tensor(ptrArray[3]), new Tensor(ptrArray[4]));
                        default: {
                                // Too long a tuple, return as a list, instead.
                                var result = new Tensor[ptrArray.Length];
                                for (var i = 0; i < ptrArray.Length; i++) {
                                    result[i] = new Tensor(ptrArray[i]);
                                }
                                return result;
                            }
                        }
                    case 3: {
                            // List of tensors
                            var result = new Tensor[ptrArray.Length];
                            for (var i = 0; i < ptrArray.Length; i++) {
                                result[i] = new Tensor(ptrArray[i]);
                            }
                            return result;
                        }
                    }

                }

                /// <summary>
                /// Invoke a function from the compilation unit.
                /// </summary>
                /// <typeparam name="TResult">The return type of the TorchScript function.</typeparam>
                /// <param name="name">The name of the function.</param>
                /// <param name="inputs">Function arguments.</param>
                public TResult invoke<TResult>(string name, params object[] inputs) => (TResult)invoke(name, inputs);

                /// <summary>
                /// Invoke a function from the compilation unit.
                /// </summary>
                /// <typeparam name="T">The type of all function arguments.</typeparam>
                /// <typeparam name="TResult">The return type of the TorchScript function.</typeparam>
                /// <param name="name">The name of the function.</param>
                /// <param name="inputs">Function arguments.</param>
                public TResult invoke<T, TResult>(string name, params T[] inputs) => (TResult)invoke(name, inputs);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSJIT_compile(string script);

            /// <summary>
            /// Create a TorchScript compilation unit containing TorchScript-compliant Python from a string.
            /// </summary>
            /// <param name="script">A string with Python code expressing a set of TorchScript functions.</param>
            /// <returns></returns>
            public static CompilationUnit compile(string script)
            {
                if (String.IsNullOrEmpty(script))
                    throw new ArgumentNullException("empty script");

                var result = THSJIT_compile(script);
                if (result == IntPtr.Zero)
                    CheckForErrors();
                return new CompilationUnit(result);
            }
        }
    }
}
