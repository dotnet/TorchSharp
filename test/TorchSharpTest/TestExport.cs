// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Runtime.InteropServices;
using Xunit;

#nullable enable

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestExport
    {
        [Fact]
        public void TestExport_LoadNonExistentFile()
        {
            Assert.Throws<ExternalException>(() =>
                torch.export.load("nonexistent.pt2"));
        }

        [Fact]
        public void TestExport_LoadInvalidFile()
        {
            var tmpFile = Path.GetTempFileName();
            try
            {
                File.WriteAllBytes(tmpFile, new byte[] { 0xDE, 0xAD, 0xBE, 0xEF });
                Assert.ThrowsAny<Exception>(() =>
                    torch.export.load(tmpFile));
            }
            finally
            {
                File.Delete(tmpFile);
            }
        }

        [Fact]
        public void TestExport_LoadEmptyPath()
        {
            Assert.ThrowsAny<Exception>(() =>
                torch.export.load(""));
        }

        [Fact]
        public void TestExport_GenericLoadNonExistentFile()
        {
            Assert.Throws<ExternalException>(() =>
                torch.export.load<torch.Tensor>("nonexistent.pt2"));
        }

        [Fact]
        public void TestExport_GenericLoadInvalidFile()
        {
            var tmpFile = Path.GetTempFileName();
            try
            {
                File.WriteAllBytes(tmpFile, new byte[] { 0xDE, 0xAD, 0xBE, 0xEF });
                Assert.ThrowsAny<Exception>(() =>
                    torch.export.load<torch.Tensor>(tmpFile));
            }
            finally
            {
                File.Delete(tmpFile);
            }
        }
    }
}
