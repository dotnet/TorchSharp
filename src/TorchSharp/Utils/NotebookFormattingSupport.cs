using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TorchSharp.Utils
{
    /// <summary>
    /// Class used to identify the formatting logic for Tensors when using .NET Interactive.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class TypeFormatterSourceAttribute : Attribute
    {
        public TypeFormatterSourceAttribute(Type formatterSourceType)
        {
            FormatterSourceType = formatterSourceType;
        }

        public Type FormatterSourceType { get; }
    }

    internal class TypeFormatterSource
    {
        public IEnumerable<object> CreateTypeFormatters()
        {
            yield return new TensorPlainTextFormatter();
            yield return new TensorHTMLFormatter();
        }
    }

    internal class TensorPlainTextFormatter
    {
        public string MimeType => "text/plain";

        public bool Format(object instance, TextWriter writer)
        {
            if (instance is not torch.Tensor result) {
                return false;
            }

            writer.Write(result.ToString(TensorStringStyle.Default));

            return true;
        }
    }

    internal class TensorHTMLFormatter
    {
        public string MimeType => "text/html";

        public bool Format(object instance, TextWriter writer)
        {
            if (instance is not torch.Tensor result) {
                return false;
            }

            writer.Write("<div><pre>" + result.ToString(TensorStringStyle.Default) + "</pre></div>");

            return true;
        }
    }
}
