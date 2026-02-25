using System;
using System.Globalization;
using TorchSharp;

class Program
{
    static void Main()
    {
        var t = torch.tensor(3.14f, torch.bfloat16);
        Console.WriteLine($"BFloat16 tensor value: {t.item<float>()}");
        Console.WriteLine($"Julia string: {t.jlstr(cultureInfo: CultureInfo.InvariantCulture)}");
        Console.WriteLine($"Numpy string: {t.npstr(cultureInfo: CultureInfo.InvariantCulture)}");
        Console.WriteLine($"CSharp string: {t.cstr(cultureInfo: CultureInfo.InvariantCulture)}");
        
        // Test a few more values to understand precision
        var t2 = torch.tensor(1.0f, torch.bfloat16);
        Console.WriteLine($"\n1.0 as bfloat16: {t2.item<float>()}");
        
        var t3 = torch.tensor(0.1f, torch.bfloat16);
        Console.WriteLine($"0.1 as bfloat16: {t3.item<float>()}");
    }
}
