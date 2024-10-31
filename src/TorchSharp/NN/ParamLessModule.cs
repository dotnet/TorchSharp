// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public interface IParameterLessModule {

        }
        /// <summary>
        /// Base class for all modules that do not have any tensor parameters or buffers, and
        /// for which the `_to()` implementation can therefore be simplified.
        /// </summary>
        public abstract class ParameterLessModule<T1, TResult> : nn.Module<T1, TResult>, IParameterLessModule
        {
            protected ParameterLessModule(string name) : base(name) { }

            protected ParameterLessModule(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) {}

            // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;

            public override void register_buffer(string name, Tensor tensor, bool persistent = true)
            {
                throw new InvalidOperationException($"Cannot register a buffer on a module that is declared 'parameter-less.'");
            }

            public override void register_parameter(string name, Parameter param)
            {
                throw new InvalidOperationException($"Cannot register a parameter on a module that is declared 'parameter-less.'");
            }

            public override void register_module(string name, nn.Module submodule)
            {
                if (submodule is not IParameterLessModule)
                    throw new InvalidOperationException($"Submodules of a parameter-less module must also be parameter-less.");
                base.register_module(name, submodule);
            }
        }

        /// <summary>
        /// Base class for all modules that do not have any tensor parameters or buffers, and
        /// for which the `_to()` implementation can therefore be simplified.
        /// </summary>
        public abstract class ParameterLessModule<T1, T2, TResult> : nn.Module<T1, T2, TResult>, IParameterLessModule
        {
            protected ParameterLessModule(string name) : base(name) { }

            protected ParameterLessModule(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) {}

            // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;


            public override void register_buffer(string name, Tensor tensor, bool persistent = true)
            {
                throw new InvalidOperationException($"Cannot register a buffer on a module that is declared 'parameter-less.'");
            }

            public override void register_parameter(string name, Parameter param)
            {
                throw new InvalidOperationException($"Cannot register a parameter on a module that is declared 'parameter-less.'");
            }

            public override void register_module(string name, nn.Module submodule)
            {
                if (submodule is not IParameterLessModule)
                    throw new InvalidOperationException($"Submodules of a parameter-less module must also be parameter-less.");
                base.register_module(name, submodule);
            }
        }

        /// <summary>
        /// Base class for all modules that do not have any tensor parameters or buffers, and
        /// for which the `_to()` implementation can therefore be simplified.
        /// </summary>
        public abstract class ParameterLessModule<T1, T2, T3, TResult> : nn.Module<T1, T2, T3, TResult>, IParameterLessModule
        {
            protected ParameterLessModule(string name) : base(name) { }

            protected ParameterLessModule(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) {}

              // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;

            public override void register_buffer(string name, Tensor tensor, bool persistent = true)
            {
                throw new InvalidOperationException($"Cannot register a buffer on a module that is declared 'parameter-less.'");
            }

            public override void register_parameter(string name, Parameter param)
            {
                throw new InvalidOperationException($"Cannot register a parameter on a module that is declared 'parameter-less.'");
            }

            public override void register_module(string name, nn.Module submodule)
            {
                if (submodule is not IParameterLessModule)
                    throw new InvalidOperationException($"Submodules of a parameter-less module must also be parameter-less.");
                base.register_module(name, submodule);
            }
        }

        /// <summary>
        /// Base class for all modules that do not have any tensor parameters or buffers, and
        /// for which the `_to()` implementation can therefore be simplified.
        /// </summary>
        public abstract class ParameterLessModule<T1, T2, T3, T4, TResult> : nn.Module<T1, T2, T3, T4, TResult>, IParameterLessModule
        {
            protected ParameterLessModule(string name) : base(name) { }

            protected ParameterLessModule(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) {}

              // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;

            public override void register_buffer(string name, Tensor tensor, bool persistent = true)
            {
                throw new InvalidOperationException($"Cannot register a buffer on a module that is declared 'parameter-less.'");
            }

            public override void register_parameter(string name, Parameter param)
            {
                throw new InvalidOperationException($"Cannot register a parameter on a module that is declared 'parameter-less.'");
            }

            public override void register_module(string name, nn.Module submodule)
            {
                if (submodule is not IParameterLessModule)
                    throw new InvalidOperationException($"Submodules of a parameter-less module must also be parameter-less.");
                base.register_module(name, submodule);
            }
        }

        /// <summary>
        /// Base class for all modules that do not have any tensor parameters or buffers, and
        /// for which the `_to()` implementation can therefore be simplified.
        /// </summary>
        public abstract class ParameterLessModule<T1, T2, T3, T4, T5, TResult> : nn.Module<T1, T2, T3, T4, T5, TResult>, IParameterLessModule
        {
            protected ParameterLessModule(string name) : base(name) { }

            protected ParameterLessModule(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) {}

              // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;

            public override void register_buffer(string name, Tensor tensor, bool persistent = true)
            {
                throw new InvalidOperationException($"Cannot register a buffer on a module that is declared 'parameter-less.'");
            }

            public override void register_parameter(string name, Parameter param)
            {
                throw new InvalidOperationException($"Cannot register a parameter on a module that is declared 'parameter-less.'");
            }

            public override void register_module(string name, nn.Module submodule)
            {
                if (submodule is not IParameterLessModule)
                    throw new InvalidOperationException($"Submodules of a parameter-less module must also be parameter-less.");
                base.register_module(name, submodule);
            }
        }

        /// <summary>
        /// Base class for all modules that do not have any tensor parameters or buffers, and
        /// for which the `_to()` implementation can therefore be simplified.
        /// </summary>
        public abstract class ParameterLessModule<T1, T2, T3, T4, T5, T6, TResult> : nn.Module<T1, T2, T3, T4, T5, T6, TResult>, IParameterLessModule
        {
            protected ParameterLessModule(string name) : base(name) { }

            protected ParameterLessModule(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) {}

              // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;

            public override void register_buffer(string name, Tensor tensor, bool persistent = true)
            {
                throw new InvalidOperationException($"Cannot register a buffer on a module that is declared 'parameter-less.'");
            }

            public override void register_parameter(string name, Parameter param)
            {
                throw new InvalidOperationException($"Cannot register a parameter on a module that is declared 'parameter-less.'");
            }

            public override void register_module(string name, nn.Module submodule)
            {
                if (submodule is not IParameterLessModule)
                    throw new InvalidOperationException($"Submodules of a parameter-less module must also be parameter-less.");
                base.register_module(name, submodule);
            }
        }
    }
}