using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp.Modules;

namespace TorchSharp.Utils
{
    public static class ModuleInfo
    {

        public class ConvInfo
        {
            public long Dimension,InChannel,OutChannel, PaddingMode;
            public object Kernel, Dilation, Stride; 
            public ConvInfo(Convolution conv)
            {
                InChannel = conv.in_channels;
                OutChannel = conv.out_channels;
                Kernel = conv.kernel_size;
                
                //TODO: Make all props;
                throw new NotImplementedException("Need finish");
            }

            public (long, long)? CastTuple(object obj)
            {
                if (obj.GetType() == typeof((long,long)))
                    return obj as (long, long)?;
                if (obj is long l)
                    return (l, l);
                return null;
            }

            public long CastValue(object obj)
            {
                var v = CastTuple(obj);
                return v?.Item1 ?? 0;
            }
        } 
    }
}
