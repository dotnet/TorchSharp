using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TorchSharp.Utils
{
    /// <summary>
    /// Specifies the custom name for a component to be used in the module's state_dict instead of the default field name.
    /// </summary>
    public class ComponentNameAttribute : Attribute
    {
        public string Name { get; set; }
    }
}
