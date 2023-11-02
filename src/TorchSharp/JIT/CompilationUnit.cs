// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using TorchSharp.PInvoke;
using static TorchSharp.PInvoke.NativeMethods;

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
                    Dispose(true);
                    GC.SuppressFinalize(this);
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

                internal IntPtr handle;

                /// <summary>
                /// Invoke a function from the compilation unit.
                /// </summary>
                /// <param name="name">The name of the function.</param>
                /// <param name="objs">Function arguments.</param>
                public object invoke(string name, params object[] objs)
                {
                    if (string.IsNullOrEmpty(name)) throw new ArgumentNullException("method name");

                    TensorOrScalar[] ptrArray = null;
                    sbyte typeCode = 0;

                    using (var ntosArray = new NativeTensorOrScalarIndexedArray()) {

                        var tRefsHandle = ScriptModule.DetermineArgumentTypeRefs(objs, out var count, ntosArray);

                        var allocated = ntosArray.Count;

                        THSJIT_CompilationUnit_Invoke(handle, name, tRefsHandle, count, ntosArray.CreateArray, out typeCode, allocated);
                        torch.CheckForErrors();
                        ptrArray = ntosArray.ToToSArray(allocated);

                        return ScriptModule.ProcessReturnValue(name, ntosArray, ptrArray, typeCode);
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

            /// <summary>
            /// Create a TorchScript compilation unit containing TorchScript-compliant Python from a string.
            /// </summary>
            /// <param name="script">A string with Python code expressing a set of TorchScript functions.</param>
            /// <returns></returns>
            public static CompilationUnit compile(string script)
            {
                if (string.IsNullOrEmpty(script))
                    throw new ArgumentNullException("empty script");

                var result = THSJIT_compile(script);
                if (result == IntPtr.Zero)
                    CheckForErrors();
                return new CompilationUnit(result);
            }
        }
    }
}
