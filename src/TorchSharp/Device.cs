using System;
using System.Collections.Generic;
using System.Text;

namespace TorchSharp
{
    public class Device
    {
        public DeviceType Type { get; private set; } = DeviceType.CPU;
        public int Index { get; private set; } = -1;

        public Device(string description)
        {
            var splits = description.Split(':');
            if (splits.Length == 1) {
                // Interpret as a device type
                Type = (DeviceType)Enum.Parse(typeof(DeviceType),splits[0].ToUpper());
            }
            else if (splits.Length == 2) {
                // Interpret as a device type and index
                Type = (DeviceType)Enum.Parse(typeof(DeviceType), splits[0].ToUpper());
                Index = int.Parse(splits[1]);
            }
        }

        public Device(DeviceType deviceType, int index = -1)
        {
            Type = deviceType;
            Index = index;
        }
    }
}
