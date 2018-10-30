using System;
using TorchSharp;

class MainClass {
	static void Dump (FloatTensor x)
	{
		for (int i = 0; i < x.GetTensorDimension (0); i++)
			Console.Write ($"{x [i]}, ");
		Console.WriteLine ();
	}

	public static void Main (string [] args)
	{
		var x = new FloatTensor (10);
		var b = new FloatTensor (10);
		b.Fill (30);
		Dump (b);
		x.Random (new RandomGenerator (), 10);
		FloatTensor.Add (x, 100, b);
		Dump (x);
		Dump (b);
#if false

		Dump (x);
		var y = x.Add (100);
		Dump (y);
#endif
		for (int i = 0; i < 1000; i++) {
			using (var a = new FloatTensor (1000)){
				using (var c = new FloatTensor (1000)) {
					var d = a.Add (10);
					a.CAdd (0, d);
					d.Dispose ();
				}
			}
		}
	}
}
