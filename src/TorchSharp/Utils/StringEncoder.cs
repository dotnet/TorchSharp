using System.Text;

namespace TorchSharp.Utils
{
    internal static class StringEncoder
    {
        internal static byte[] GetNullTerminatedUTF8ByteArray(string input)
        {
            var bytes = new byte[Encoding.UTF8.GetMaxByteCount(input.Length)];
            var len = Encoding.UTF8.GetBytes(input, 0, input.Length, bytes, 0);
            bytes[len] = 0;
            return bytes;
        }
    }
}
