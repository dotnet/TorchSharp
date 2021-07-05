using System;
using System.Collections.Generic;
using System.Text;

namespace TorchSharp
{
    public static partial class torch
    {
        /// <summary>
        /// A torch.Device is an object representing the device on which a torch.Tensor is or will be allocated.
        /// </summary>
        public class Device
        {
            public DeviceType type { get; private set; } = DeviceType.CPU;
            public int index { get; private set; } = -1;

            public Device(string description)
            {
                var splits = description.Split(':');
                if (splits.Length == 1) {
                    // Interpret as a device type
                    type = (DeviceType)Enum.Parse(typeof(DeviceType), splits[0].ToUpper());
                } else if (splits.Length == 2) {
                    // Interpret as a device type and index
                    type = (DeviceType)Enum.Parse(typeof(DeviceType), splits[0].ToUpper());
                    index = int.Parse(splits[1]);
                }
            }

            public Device(string deviceType, int index = -1)
            {
                type = (DeviceType)Enum.Parse(typeof(DeviceType), deviceType.ToUpper());
                this.index = index;
            }
            public Device(DeviceType deviceType, int index = -1)
            {
                type = deviceType;
                this.index = index;
            }

            public Device(int index)
            {
                type = DeviceType.CUDA;
                this.index = index;
            }

            public override string ToString()
            {
                return type == DeviceType.CPU ? "cpu" : (index == -1) ? $"{type.ToString().ToLower()}" : $"{type.ToString().ToLower()}:{index}";
            }

            public static implicit operator Device(string description)
            {
                return new Device(description);
            }
        }

        /// <summary>
        /// Convenience declaration of a CPU device accessible everywhere.
        /// </summary>
        public static Device CPU = new Device(DeviceType.CPU, -1);

        /// <summary>
        /// Convenience declaration of a CPU device accessible everywhere.
        /// </summary>
        public static Device CUDA = new Device(DeviceType.CUDA, -1);

        /// <summary>
        /// Factory for a device object, following the Pytorch API.
        /// </summary>
        /// <param name="description">String description of the device, e.g. 'cpu' or 'cuda:0'</param>
        /// <returns></returns>
        public static Device device(string description) => new Device(description);

        /// <summary>
        /// Factory for a device object, following the Pytorch API.
        /// </summary>
        /// <param name="deviceType">The device type, e.g. 'cpu' or 'cuda'</param>
        /// <param name="index">The device index. Ignored for CPUs</param>
        /// <returns></returns>
        public static Device device(string deviceType, int index = -1) => new Device(deviceType, index);

        /// <summary>
        /// Factory for a device object, following the Pytorch API.
        /// </summary>
        /// <param name="deviceType">The device type, e.g. DeviceType.CPU or DeviceType.CUDA.</param>
        /// <param name="index">The device index. Ignored for CPUs</param>
        /// <returns></returns>
        public static Device device(DeviceType deviceType, int index = -1) => new Device(deviceType, index);

        /// <summary>
        /// Factory for a CUDA device object, following the Pytorch API.
        /// </summary>
        /// <param name="index">The CUDA device ordinal.</param>
        /// <returns></returns>
        public static Device device(int index) => new Device(DeviceType.CUDA, index);
    }
}
