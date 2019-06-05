using System;
using System.Runtime.InteropServices;

namespace AtenSharp {
	/// <summary>
	/// Random class
	/// </summary>
	/// <remarks>
	/// Behind the scenes this is the THGenerator API and THRandom combined into
	/// one as THRandom are just convenience methods on top of THGenerator.
	/// </remarks>
	public class RandomGenerator : IDisposable {
		internal IntPtr handle;

		[DllImport ("caffe2")]
		extern static void THGenerator_free (IntPtr handle);

		[DllImport ("caffe2")]
		extern static IntPtr THGenerator_new ();

		/// <summary>
		/// Initializes a new instance of the <see cref="T:AtenSharp.RandomGenerator"/> class.
		/// </summary>
		public RandomGenerator ()
		{
			handle = THGenerator_new ();
		}

		[DllImport ("caffe2")]
		extern static int THGeneratorState_isValid (IntPtr handle);

		/// <summary>
		/// Gets a value indicating whether this <see cref="T:AtenSharp.RandomGenerator"/> is valid.
		/// </summary>
		/// <value><c>true</c> if is valid; otherwise, <c>false</c>.</value>
		public bool IsValid => THGeneratorState_isValid (handle) != 0;

		/// <summary>
		/// Dispose the specified disposing.
		/// </summary>
		/// <param name="disposing">If set to <c>true</c> disposing.</param>
		protected virtual void Dispose (bool disposing)
		{
			if (handle != IntPtr.Zero) {
				THGenerator_free (handle);
				handle = IntPtr.Zero;
			}
		}

		/// <summary>
		/// Releases unmanaged resources and performs other cleanup operations before the
		/// <see cref="T:AtenSharp.RandomGenerator"/> is reclaimed by garbage collection.
		/// </summary>
		~RandomGenerator()
		{
			Dispose (false);
		}

		/// <summary>
		/// Releases all resource used by the <see cref="T:AtenSharp.RandomGenerator"/> object.
		/// </summary>
		/// <remarks>Call Dispose when you are finished using the <see cref="T:AtenSharp.RandomGenerator"/>. This
		/// method leaves the <see cref="T:AtenSharp.RandomGenerator"/> in an unusable state. After
		/// calling this method, you must release all references to the <see cref="T:AtenSharp.RandomGenerator"/>
		/// so the garbage collector can reclaim the memory that the <see cref="T:AtenSharp.RandomGenerator"/> was occupying.</remarks>
		public void Dispose ()
		{
			Dispose (true);
			GC.SuppressFinalize (this);
		}

		[DllImport ("caffe2")]
		extern static ulong THRandom_seed (IntPtr handle);

		/// <summary>
		/// Initializes the random number generator from /dev/urandom or in Windows with the current time.
		/// </summary>
		/// <returns>The random seed.</returns>
		public ulong InitRandomSeed () => THRandom_seed (handle);

		[DllImport ("caffe2")]
		extern static void THRandom_manualSeed (IntPtr handle, ulong seed);

		/// <summary>
		/// Initializes the random number generator with the given seed.
		/// </summary>
		/// <param name="seed">Seed.</param>
		public void InitWithSeed (ulong seed) => THRandom_manualSeed (handle, seed);


		[DllImport ("caffe2")]
		extern static ulong THRandom_initialSeed (IntPtr handle);

		/// <summary>
		///  Returns the starting seed used.
		/// </summary>
		/// <value>The initial seed.</value>
		public ulong InitialSeed => THRandom_initialSeed (handle);

		[DllImport ("caffe2")]
		extern static ulong THRandom_random (IntPtr handle);

		/// <summary>
		///  Generates a uniform 32 bits integer. 
		/// </summary>
		/// <returns>UInt32 random value.</returns>
		public uint NextUInt32 () => (uint)THRandom_random (handle);

		[DllImport ("caffe2")]
		extern static ulong THRandom_random64 (IntPtr handle);

		/// <summary>
		///  Generates a uniform 64 bits integer. 
		/// </summary>
		/// <returns>UInt64 random value.</returns>
		public ulong NextUInt64 () => THRandom_random64 (handle);

		[DllImport ("caffe2")]
		extern static double THRandom_standard_uniform (IntPtr handle);

		/// <summary>
		///  Generates a uniform random double on [0,1).
		/// </summary>
		/// <returns>Generates a uniform random double on [0,1).</returns>
		public double NextDouble () => THRandom_standard_uniform (handle);

		[DllImport ("caffe2")]
		extern static double THRandom_uniform (IntPtr handle, double a, double b);

		/// <summary>
		///  Generates a uniform random double on [a,b).
		/// </summary>
		/// <returns>Generates a uniform random double on [a, b).</returns>
		public double NextDouble (double a, double b) => THRandom_uniform (handle, a, b);

		[DllImport ("caffe2")]
		extern static float THRandom_uniformFloat (IntPtr handle, float a, float b);

		/// <summary>
		///  Generates a uniform random float on [a,b).
		/// </summary>
		/// <returns>Generates a uniform random float on [a, b).</returns>
		public double NextFloat (float a, float b) => THRandom_uniformFloat (handle, a, b);

		[DllImport ("caffe2")]
		extern static double THRandom_normal (IntPtr handle, double mean, double stddev);

		/// <summary>
		///  Generates a random number from a normal distribution.
		/// </summary>
		/// <param name="mean">Mean for the distribution</param>
		/// <param name="stdev">Stanard deviation for the distribution, > 0 </param>
		/// <returns>Generates a number for the normal distribution.</returns>
		public double NextNormalDouble (double mean, double stdev) => THRandom_normal (handle, mean, stdev);

		[DllImport ("caffe2")]
		extern static double THRandom_exponential (IntPtr handle, double lambda);

		/// <summary>
		///  Generates a random number from an exponential distribution.
		/// </summary>
		/// <param name="lambda">Must be a positive number</param>
		/// <remarks>
		/// The density is $p(x) = lambda * exp(-lambda * x)$, where lambda is a positive number.
		/// </remarks>
		public double NextExponentialDouble (double lambda) => THRandom_exponential (handle, lambda);

		[DllImport ("caffe2")]
		extern static double THRandom_cauchy (IntPtr handle, double median, double sigma);

		/// <summary>
		///  Returns a random number from a Cauchy distribution.
		/// </summary>
		/// <param name="median"></param>
		/// <param name="sigma"></param>
		/// <remarks>
		/// The Cauchy density is $p(x) = sigma/(pi*(sigma^2 + (x-median)^2))$
		/// </remarks>
		public double NextCauchyDouble (double median, double sigma) => THRandom_cauchy (handle, median, sigma);

		[DllImport ("caffe2")]
		extern static double THRandom_logNormal (IntPtr handle, double mean, double stddev);

		/// <summary>
		///  Generates a random number from a log-normal distribution.
		/// </summary>
		/// <param name="mean">&gt; 0 is the mean of the log-normal distribution</param>
		/// <param name="stddev">is its standard deviation.</param>
		public double NextLogNormalDouble (double mean, double stddev) => THRandom_logNormal (handle, mean, stddev);

		[DllImport ("caffe2")]
		extern static double THRandom_geometric (IntPtr handle, double p);

		/// <summary>
		///  Generates a random number from a geometric distribution.
		/// </summary>
		/// <remarks>
		/// It returns an integer i, where p(i) = (1-p) * p^(i-1).
		/// p must satisfy $0 &lt; p &lt; 1
		/// </remarks>
		public double NextGeometricDouble (double mean, double p) => THRandom_geometric (handle, p);

		[DllImport ("caffe2")]
		extern static double THRandom_bernoulli (IntPtr handle, double p);

		/// <summary>
		///  Returns true with double probability $p$ and false with probability 1-p (p &gt; 0).
		/// </summary>
		public double NextBernoulliDouble (double mean, double p) => THRandom_bernoulli (handle, p);

		[DllImport ("caffe2")]
		extern static float THRandom_bernoulliFloat (IntPtr handle, float p);

		/// <summary>
		///  Returns true with double probability $p$ and false with probability 1-p (p &gt; 0).
		/// </summary>
		public float NextBernoulliDouble (double mean, float p) => THRandom_bernoulliFloat (handle, p);

	}
}