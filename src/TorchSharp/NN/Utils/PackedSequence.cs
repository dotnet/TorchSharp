// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class nn
        {
            public static partial class utils
            {
                public static partial class rnn
                {
                    /// <summary>
                    /// A packed batch of variable length sequences.
                    /// </summary>
                    public sealed class PackedSequence : IDisposable
                    {
                        /// <summary>
                        /// Class wrapping PyTorch's packedsequence object reference.
                        /// </summary>
                        internal sealed class HType : SafeHandle
                        {
                            public HType(IntPtr preexistingHandle, bool ownsHandle)
                                : base(IntPtr.Zero, ownsHandle)
                            {
                                SetHandle(preexistingHandle);
                            }

                            public override bool IsInvalid => handle == IntPtr.Zero;

                            // This is just for marshalling
                            internal HType() : base(IntPtr.Zero, true)
                            {
                            }

                            [DllImport("LibTorchSharp")]
                            private static extern void THSNN_PackedSequence_dispose(IntPtr handle);

                            protected override bool ReleaseHandle()
                            {
                                THSNN_PackedSequence_dispose(handle);
                                return true;
                            }
                        }

                        private HType handle;

                        internal PackedSequence(HType handle)
                        {
                            this.handle = handle;
                        }

                        internal HType Handle => handle;

                        /// <summary>
                        ///   Releases the storage.
                        /// </summary>
                        public void Dispose()
                        {
                            if (handle != null && !handle.IsInvalid) {
                                handle.Dispose();
                                handle.SetHandleAsInvalid();
                            }
                        }
                    }
                }
            }
        }
    }
}
