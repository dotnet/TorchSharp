using System;
using System.Globalization;
using TorchSharp;

var t = torch.tensor(3.14f, torch.bfloat16);
Console.WriteLine($"BFloat16 tensor value: {t.item<float>()}");
Console.WriteLine($"Julia string: {t.jlstr(cultureInfo: CultureInfo.InvariantCulture)}");
Console.WriteLine($"Numpy string: {t.npstr(cultureInfo: CultureInfo.InvariantCulture)}");
Console.WriteLine($"CSharp string: {t.cstr(cultureInfo: CultureInfo.InvariantCulture)}");
