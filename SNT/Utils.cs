using System;

namespace Torch.SNT
{
    internal class Utils
    {
        public static int GetTotalLength(ReadOnlySpan<int> dimensions, int startIndex = 0)
        {
            if (dimensions.Length == 0)
            {
                return 0;
            }

            int product = 1;
            for (int i = startIndex; i < dimensions.Length; i++)
            {
                if (dimensions[i] < 0)
                {
                    throw new ArgumentOutOfRangeException($"{nameof(dimensions)}[{i}]");
                }

                checked
                {
                    product *= dimensions[i];
                }
            }

            return product;
        }
    }
}
