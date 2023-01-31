#nullable enable
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;

namespace TorchSharp
{
    /// <summary>
    /// To ignore an xUnit <see cref="FactAttribute">Fact</see> on a given platform or architecture.
    /// </summary>
    public sealed class FactIgnoreOnPlatformAttribute : FactAttribute
    {
        public FactIgnoreOnPlatformAttribute(string skip, params string[] platforms)
        {
            if (platforms.Any(p => RuntimeInformation.IsOSPlatform(OSPlatform.Create(p.ToUpperInvariant())))) {
                Skip = skip;
            }
        }

        public FactIgnoreOnPlatformAttribute(params string[] platforms)
        {
            if (platforms.Any(p => RuntimeInformation.IsOSPlatform(OSPlatform.Create(p.ToUpperInvariant())))) {
                Skip = $"based on platform {RuntimeInformation.OSDescription}";
            }
        }

        public FactIgnoreOnPlatformAttribute(string skip, string platform, Architecture architecture)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Create(platform.ToUpperInvariant()))
                && RuntimeInformation.ProcessArchitecture == architecture) {
                Skip = skip;
            }
        }

        public FactIgnoreOnPlatformAttribute(string platform, Architecture architecture)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Create(platform.ToUpperInvariant()))
                && RuntimeInformation.ProcessArchitecture == architecture) {
                Skip = $"based on platform {platform} {architecture}";
            }
        }
    }
}