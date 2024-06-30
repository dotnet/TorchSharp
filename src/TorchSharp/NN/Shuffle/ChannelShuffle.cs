// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a ChannelShuffle module.
        /// </summary>
        public sealed class ChannelShuffle : torch.nn.Module<Tensor, Tensor>
        {
            internal ChannelShuffle(long groups) : base(nameof(ChannelShuffle))
            {
                this.groups = groups;
            }
            private long groups;

            public override Tensor forward(Tensor tensor)
            {
                return tensor.channel_shuffle(groups);
            }

            public override string GetName()
            {
                return typeof(ChannelShuffle).Name;
            }

            // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Divide the channels in a tensor into g groups and rearrange them.
            ///
            /// See: https://pytorch.org/docs/1.10/generated/torch.nn.ChannelShuffle.html#channelshuffle
            /// </summary>
            /// <param name="groups">The number of groups to divide channels in.</param>
            /// <returns></returns>
            public static ChannelShuffle ChannelShuffle(long groups)
            {
                return new ChannelShuffle(groups);
            }
        }
    }
}
