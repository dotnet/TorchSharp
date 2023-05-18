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
        /// Base class for all modules that do not have any parameters or buffers, and
        /// for which the `_to()` implementation can therefore be simplified.
        /// </summary>
        public abstract class ParamLessModule<T1, T2> : nn.Module<T1, T2>
        {
            protected ParamLessModule(string name) : base(name) { }

            // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype) => this;

            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex = -1) => this;

            protected internal override nn.Module _to(ScalarType dtype) => this;
        }

        /// <summary>
        /// Base class for all modules that do not have any parameters or buffers, and
        /// for which the `_to()` implementation can therefore be simplified.
        /// </summary>
        public abstract class ParamLessModule<T1, T2, T3> : nn.Module<T1, T2, T3>
        {
            protected ParamLessModule(string name) : base(name) { }

            // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype) => this;

            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex = -1) => this;

            protected internal override nn.Module _to(ScalarType dtype) => this;
        }
    }
}