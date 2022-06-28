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
                            private static extern void THSNN_PackedSequence_dispose(HType handle);

                            protected override bool ReleaseHandle()
                            {
                                THSNN_PackedSequence_dispose(this);
                                return true;
                            }

                            protected override void Dispose(bool disposing)
                            {
                                if (disposing) {
                                    ReleaseHandle();
                                }
                            }
                        }

                        private HType handle;

                        internal PackedSequence(IntPtr handle)
                        {
                            if (handle != IntPtr.Zero) {
                                this.handle = new HType(handle, true);
                            }
                        }

                        ~PackedSequence()
                        {
                            Dispose(false);
                        }

                        internal HType Handle => handle;

                        /// <summary>
                        ///   Releases the storage.
                        /// </summary>
                        public void Dispose()
                        {
                            Dispose(true);
                            GC.SuppressFinalize(this);
                        }

                        /// <summary>
                        ///   Implements the .NET Dispose pattern.
                        /// </summary>
                        protected virtual void Dispose(bool disposing)
                        {
                            if (disposing && handle != null && !handle.IsInvalid) {
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
