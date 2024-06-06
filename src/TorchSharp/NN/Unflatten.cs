// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent an unflattening operation.
        /// </summary>
        public sealed class Unflatten : ParamLessModule<Tensor, Tensor>
        {
            internal Unflatten(long dim, long[] unflattened_size) : base(nameof(Unflatten))
            {
                this.dim = dim;
                this.unflattened_size = unflattened_size;
            }

            public override Tensor forward(Tensor tensor)
            {
                return tensor.unflatten(dim, unflattened_size);
            }

            // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;
            
            public long dim { get; set; }
            public long[] unflattened_size { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Unflattens a tensor dim expanding it to a desired shape. For use with Sequential.
            /// </summary>
            /// <param name="dim">Dimension to be unflattened</param>
            /// <param name="unflattened_size">New shape of the unflattened dimension</param>
            /// <returns></returns>
            public static Unflatten Unflatten(long dim, long[] unflattened_size)
            {
                return new Unflatten(dim, unflattened_size);
            }
        }
    }
}
