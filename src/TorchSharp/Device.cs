using System;
using System.Collections.Generic;
using System.Text;

namespace TorchSharp
{
    public static partial class torch
    {
        public class device
        {
            public DeviceType Type { get; private set; } = DeviceType.CPU;
            public int Index { get; private set; } = -1;

            public static device CPU = new device(DeviceType.CPU, -1);
            public static device CUDA = new device(DeviceType.CUDA, -1);

            public device(string description)
            {
                var splits = description.Split(':');
                if (splits.Length == 1) {
                    // Interpret as a device type
                    Type = (DeviceType)Enum.Parse(typeof(DeviceType), splits[0].ToUpper());
                } else if (splits.Length == 2) {
                    // Interpret as a device type and index
                    Type = (DeviceType)Enum.Parse(typeof(DeviceType), splits[0].ToUpper());
                    Index = int.Parse(splits[1]);
                }
            }

            public device(string deviceType, int index = -1)
            {
                Type = (DeviceType)Enum.Parse(typeof(DeviceType), deviceType.ToUpper());
                Index = index;
            }
            public device(DeviceType deviceType, int index = -1)
            {
                Type = deviceType;
                Index = index;
            }

            public override string ToString()
            {
                return Type == DeviceType.CPU ? "cpu" : (Index == -1) ? $"{Type.ToString().ToLower()}" : $"{Type.ToString().ToLower()}:{Index}";
            }
        }
    }
}
