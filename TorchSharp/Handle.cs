using Microsoft.Win32.SafeHandles;
using System;

namespace TorchSharp {
	internal sealed class TorchHandle : SafeHandleZeroOrMinusOneIsInvalid {
		public TorchHandle (IntPtr preexistingHandle, bool ownsHandle) : base (ownsHandle)
		{
			SetHandle (preexistingHandle);
		}

		// This is just for marshalling
		internal TorchHandle () : base (true)
		{
		}

		protected override bool ReleaseHandle ()
		{
			// ouch.   
			// Will likely need to abandon this attempt.
			return true;
		}

	}
}