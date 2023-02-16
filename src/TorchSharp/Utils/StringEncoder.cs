using System.Text;

namespace TorchSharp.Utils
{
    internal static class StringEncoder
    {
        private static readonly Encoding s_utfEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: false);

        internal static byte[] GetNullTerminatedUTF8ByteArray(string input)
        {
            var bytes = new byte[s_utfEncoding.GetMaxByteCount(input.Length)+1];
            var len = s_utfEncoding.GetBytes(input, 0, input.Length, bytes, 0);
            bytes[len] = 0;
            return bytes;
        }
    }
}
