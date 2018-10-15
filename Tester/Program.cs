using System;
using TorchSharp;

class MainClass {
	public static void Main (string [] args)
	{
		var x = new FloatTensor (10);
		x.Random (new THRandom (), 10);

	}
}
