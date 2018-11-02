using System;
using TorchSharp;

class MainClass {
	static void Dump (FloatTensor x, string message = "")
	{
		if (!string.IsNullOrEmpty(message)) Console.Write($"{message}(f): ");
		for (int i = 0; i < x.GetTensorDimension (0); i++)
			Console.Write ($"{x [i]}, ");
		Console.WriteLine ();
	}

	static void Dump (ByteTensor x, string message = "")
	{
		if (!string.IsNullOrEmpty(message)) Console.Write($"{message}(b): ");
		for (int i = 0; i < x.GetTensorDimension (0); i++)
			Console.Write ($"{x [i]}, ");
		Console.WriteLine ();
	}

	static void Dump (long [] x, string message = "")
	{
		if (!string.IsNullOrEmpty(message)) Console.Write($"{message}: ");
		for (int i = 0; i < x.Length; i++)
			Console.Write ($"{x [i]}, ");
		Console.WriteLine ();
	}

	public static void Main (string [] args)
	{
		var x = new FloatTensor (10);
		var a = new FloatTensor (10);
		var b = new FloatTensor (10);
		var c = new FloatTensor (10);

		var idx = new LongTensor(3);
		idx[0] = 1;
		idx[1] = 5;
		idx[2] = 8;

		c = a.OnesLike();
		b = a.OnesLike();
		Dump(b, nameof(b));
		c = FloatTensor.ARange(1.23,17.0,1.5);
		Dump (c, nameof(c));

		var bytes = new ByteTensor (10);
			
		for (var i = 0; i < 10; ++i)
		{
			a[i] = 1;
			b[i] = i+1;
			c[i] = 100*i;
			bytes[i] = (byte)(10*i);
		}

		Dump (b, nameof(b));
		Dump (c, nameof(c));

		var e = c.Concatenate(b,0);
		Dump(e, nameof(e));

		Dump (a, nameof(a));
		Dump (b, nameof(b));
		Dump (c, nameof(c));

		x.LogNormal (new RandomGenerator (), 5.5, 1.05);

		var d = a.LtTensorT(b);
		Dump (d, "d");

		FloatTensor.Add (x, 100, b);
		Dump (x, "x");
		Dump (b, "b");

		var h = x.Histc(5,20,820);
		Dump (h, nameof(h));
#if false

		Dump (x);
		var y = x.Add (100);
		Dump (y);
#endif
		for (int i = 0; i < 1000; i++) {
			using (var t1 = new FloatTensor (1000)){
				using (var t2 = new FloatTensor (1000)) {
					var t3 = t1.Add (10);
					t1.CAdd (0, t3);
					t3.Dispose ();
				}
			}
		}
	}
}
