using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TorchSharp.Utils
{
    public class ComponentNameAttribute : Attribute
    {
        public string Name { get; set; }

        public ComponentNameAttribute(string name)
        {
            Name = name;
        }
    }
}
