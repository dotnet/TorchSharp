using System;
using System.Runtime.InteropServices;

namespace TorchSharp {
	/// <summary>
	/// Random class
	/// </summary>
    	/// <remarks>
	/// Behind the scenes this is the THGenerator API and THRandom combined into
    	/// one as THRandom are just convenience methods on top of THGenerator.
    	/// </remarks>
	public class THRandom : IDisposable {
		internal IntPtr handle;

		[DllImport ("caffe2")]
		extern static void THGenerator_free (IntPtr handle);

		[DllImport ("caffe2")]
		extern static IntPtr THGenerator_new ();

		public THRandom ()
		{
			handle = THGenerator_new ();
		}

		[DllImport ("caffe2")]
		extern static int THGeneratorState_isValid (IntPtr handle);

		public bool IsValid => THGeneratorState_isValid (handle) != 0;

		protected virtual void Dispose (bool disposing)
		{
			if (handle != IntPtr.Zero) {
				THGenerator_free (handle);
				handle = IntPtr.Zero;
			}
		}

		~THRandom ()
		{
			Dispose (false);
		}

		// This code added to correctly implement the disposable pattern.
		public void Dispose ()
		{
			Dispose (true);
			GC.SuppressFinalize(this);
		}
	}
}
