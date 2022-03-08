using System.IO;
using Xunit;

namespace TorchSharp
{
    public class NetStandardTests
    {
        [Theory]
        [InlineData(@"c:\users\me", @"middle", @"c:\users\me\middle")]
        [InlineData(@"c:\users\me", null, @"c:\users\me")]
        [InlineData(null, @"last", @"last")]
        [InlineData(@"c:\users\me", "", @"c:\users\me")]
        [InlineData("", @"last", @"last")]
        [InlineData(@"c:\users\me\", @"middle\", @"c:\users\me\middle\")]
        [InlineData(@"c:\users\me", @"\middle\", @"c:\users\me\middle\")]
        public static void TestNSPath2Parts(string s1, string s2, string expected) => Assert.Equal(expected, NSPath.Join(s1, s2));

        [Theory]
        [InlineData(@"c:\users\me", @"middle", @"last", @"c:\users\me\middle\last")]
        [InlineData(null, @"middle", @"last", @"middle\last")]
        [InlineData(@"c:\users\me", null, @"last", @"c:\users\me\last")]
        [InlineData(@"c:\users\me", @"middle", null, @"c:\users\me\middle")]
        [InlineData(@"c:\users\me\", @"middle", @"last", @"c:\users\me\middle\last")]
        [InlineData(@"c:\users\me", @"\middle\", @"last", @"c:\users\me\middle\last")]
        [InlineData(@"c:\users\me", @"\middle", @"\last", @"c:\users\me\middle\last")]
        public static void TestNSPath3Parts(string s1, string s2, string s3, string expected) => Assert.Equal(expected, NSPath.Join(s1, s2, s3));

        [Theory]
        [InlineData(@"c:\users\me", @"middle1", @"middle2", @"last", @"c:\users\me\middle1\middle2\last")]
        [InlineData(null, @"middle1", @"middle2", @"last", @"middle1\middle2\last")]
        [InlineData(@"c:\users\me", null, @"middle2", @"last", @"c:\users\me\middle2\last")]
        [InlineData(@"c:\users\me", @"middle1", null, @"last", @"c:\users\me\middle1\last")]
        [InlineData(@"c:\users\me", @"middle1", @"middle2", null, @"c:\users\me\middle1\middle2")]
        [InlineData(@"c:\users\me\", @"middle1\", @"middle2\", @"last", @"c:\users\me\middle1\middle2\last")]
        [InlineData(@"c:\users\me", @"\middle1", @"\middle2", @"\last", @"c:\users\me\middle1\middle2\last")]
        public static void TestNSPath4Parts(string s1, string s2, string s3, string s4, string expected) => Assert.Equal(expected, NSPath.Join(s1, s2, s3, s4));
    }
}
