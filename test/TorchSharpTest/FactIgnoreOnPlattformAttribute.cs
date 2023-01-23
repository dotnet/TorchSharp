#nullable enable
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;

namespace TorchSharp
{
    /// <summary>
    /// To ignore an xUnit <see cref="FactAttribute">Fact</see> on a given platform or architecture.
    /// </summary>
    public sealed class FactIgnoreOnPlattformAttribute : FactAttribute
    {
        public FactIgnoreOnPlattformAttribute(string skip, params string[] plattforms)
        {
            if (plattforms.Any(p => RuntimeInformation.IsOSPlatform(OSPlatform.Create(p.ToUpperInvariant())))) {
                Skip = skip;
            }
        }

        public FactIgnoreOnPlattformAttribute(params string[] plattforms)
        {
            if (plattforms.Any(p => RuntimeInformation.IsOSPlatform(OSPlatform.Create(p.ToUpperInvariant())))) {
                Skip = $"based on platform {RuntimeInformation.OSDescription}";
            }
        }

        public FactIgnoreOnPlattformAttribute(string skip, string plattform, Architecture architecture)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Create(plattform.ToUpperInvariant()))
                && RuntimeInformation.ProcessArchitecture == architecture) {
                Skip = skip;
            }
        }

        public FactIgnoreOnPlattformAttribute(string plattform, Architecture architecture)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Create(plattform.ToUpperInvariant()))
                && RuntimeInformation.ProcessArchitecture == architecture) {
                Skip = $"based on platform {plattform} {architecture}";
            }
        }
    }
}