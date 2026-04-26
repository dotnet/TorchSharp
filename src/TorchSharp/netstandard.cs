// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

//
// This file contains the implementation of functionality exist in .NET and not supported in netstandard.
// The implementation is supported on Windows only. The goal mainly is to support .NET Framework.
//

using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace System
{
    internal static partial class NSExtension
    {
        public static HashSet<TSource> ToHashSet<TSource>(this IEnumerable<TSource> source) => source.ToHashSet(comparer: null);

        public static HashSet<TSource> ToHashSet<TSource>(this IEnumerable<TSource> source, IEqualityComparer<TSource> comparer)
        {
            if (source == null)
            {
                throw new ArgumentNullException(nameof(source));
            }

            // Don't pre-allocate based on knowledge of size, as potentially many elements will be dropped.
            return new HashSet<TSource>(source, comparer);
        }

        public static bool TryAdd<TKey, TValue>(this IDictionary<TKey, TValue> dictionary, TKey key, TValue value)
        {
            if (!dictionary.TryGetValue(key, out _))
            {
                dictionary.Add(key, value);
                return true;
            }

            return false;
        }

        public static bool Contains(this string source, char c) => !string.IsNullOrEmpty(source) && source.IndexOf(c) >= 0;
        public static bool EndsWith(this string source, char c) => !string.IsNullOrEmpty(source) && source[source.Length - 1] == c;
        public static bool StartsWith(this string source, char c) => !string.IsNullOrEmpty(source) && source[0] == c;
    }

    // MathF emulation on platforms which don't support it natively.
    internal static class MathF
    {
        public const float PI = (float)Math.PI;

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Max(float x, float y) => Math.Max(x, y);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Round(float x) => (float)Math.Round(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Sqrt(float x) => (float)Math.Sqrt(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int Sign(float x) => Math.Sign(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Abs(float x) => Math.Abs(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Log10(float x) => log10f(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Log2(float x) => log2f(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Acos(float x) => acosf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Cos(float x) => cosf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Cosh(float x) => coshf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Atan(float x) => atanf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Tanh(float x) => tanhf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Pow(float x, float y) => powf(x, y);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Sin(float x) => sinf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Sinh(float x) => sinhf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Asinh(float x) => asinhf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Tan(float x) => tanf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Acosh(float x) => acoshf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Atanh(float x) => atanhf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Asin(float x) => asinf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Log(float x) => logf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Exp(float x) => expf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Floor(float x) => floorf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Ceiling(float x) => ceilf(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Truncate(float x) => truncf(x);

        public static float Log(float x, float y)
        {
            if (float.IsNaN(x))
            {
                return x; // IEEE 754-2008: NaN payload must be preserved
            }

            if (float.IsNaN(y))
            {
                return y; // IEEE 754-2008: NaN payload must be preserved
            }

            if (y == 1)
            {
                return float.NaN;
            }

            if ((x != 1) && ((y == 0) || float.IsPositiveInfinity(y)))
            {
                return float.NaN;
            }

            return Log(x) / Log(y);
        }

        public static float IEEERemainder(float x, float y)
        {
            if (float.IsNaN(x))
            {
                return x; // IEEE 754-2008: NaN payload must be preserved
            }

            if (float.IsNaN(y))
            {
                return y; // IEEE 754-2008: NaN payload must be preserved
            }

            float regularMod = x % y;

            if (float.IsNaN(regularMod))
            {
                return float.NaN;
            }

            if ((regularMod == 0) && IsNegative(x))
            {
                return NegativeZero;
            }

            float alternativeResult = (regularMod - (Abs(y) * Sign(x)));

            if (Abs(alternativeResult) == Abs(regularMod))
            {
                float divisionResult = x / y;
                float roundedResult = Round(divisionResult);

                if (Abs(roundedResult) > Abs(divisionResult))
                {
                    return alternativeResult;
                }
                else
                {
                    return regularMod;
                }
            }

            if (Abs(alternativeResult) < Abs(regularMod))
            {
                return alternativeResult;
            }
            else
            {
                return regularMod;
            }
        }

        internal const float NegativeZero = (float)-0.0;
        internal static bool IsNegative(float x) => (((ulong)BitConverter.DoubleToInt64Bits(x)) & 0x8000000000000000) == 0x8000000000000000;

        private const string CrtLibrary = "ucrtbase.dll";
        [DllImport(CrtLibrary)] private static extern float asinhf(float x);
        [DllImport(CrtLibrary)] private static extern float acosf(float x);
        [DllImport(CrtLibrary)] private static extern float cosf(float x);
        [DllImport(CrtLibrary)] private static extern float coshf(float x);
        [DllImport(CrtLibrary)] private static extern float atanf(float x);
        [DllImport(CrtLibrary)] private static extern float tanhf(float x);
        [DllImport(CrtLibrary)] private static extern float powf(float x, float y);
        [DllImport(CrtLibrary)] private static extern float sinf(float x);
        [DllImport(CrtLibrary)] private static extern float sinhf(float x);
        [DllImport(CrtLibrary)] private static extern float tanf(float x);
        [DllImport(CrtLibrary)] private static extern float logf(float x);
        [DllImport(CrtLibrary)] private static extern float acoshf(float x);
        [DllImport(CrtLibrary)] private static extern float atanhf(float x);
        [DllImport(CrtLibrary)] private static extern float asinf(float x);
        [DllImport(CrtLibrary)] private static extern float log10f(float x);
        [DllImport(CrtLibrary)] private static extern float log2f(float x);
        [DllImport(CrtLibrary)] private static extern float expf(float x);
        [DllImport(CrtLibrary)] private static extern float floorf(float x);
        [DllImport(CrtLibrary)] private static extern float ceilf(float x);
        [DllImport(CrtLibrary)] private static extern float truncf(float x);
    }

    internal static partial class NativeLibrary
    {
        [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError = true)]
        internal static extern IntPtr LoadLibraryEx(string lpFileName, IntPtr hReservedNull, int dwFlags);

        public static bool TryLoad(string libraryPath, out IntPtr handle)
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                throw new NotSupportedException("This call is not supported on non-Windows platforms.");
            }

            if (libraryPath == null)
                throw new ArgumentNullException(nameof(libraryPath));

            handle = LoadFromPath(libraryPath, throwOnError: false);
            return handle != IntPtr.Zero;
        }

        public static bool TryLoad(string libraryName, Assembly assembly, DllImportSearchPath? searchPath, out IntPtr handle)
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                throw new NotSupportedException("This call is not supported on non-Windows platforms.");
            }

            if (libraryName == null)
                throw new ArgumentNullException(nameof(libraryName));
            if (assembly == null)
                throw new ArgumentNullException(nameof(assembly));

            handle = LoadLibraryByName(libraryName,
                                assembly,
                                searchPath,
                                throwOnError: false);
            return handle != IntPtr.Zero;
        }

        internal static IntPtr LoadLibraryByName(string libraryName, Assembly assembly, DllImportSearchPath? searchPath, bool throwOnError)
        {
            // First checks if a default dllImportSearchPathFlags was passed in, if so, use that value.
            // Otherwise checks if the assembly has the DefaultDllImportSearchPathsAttribute attribute.
            // If so, use that value.

            int searchPathFlags;
            bool searchAssemblyDirectory;
            if (searchPath.HasValue)
            {
                searchPathFlags = (int)(searchPath.Value & ~DllImportSearchPath.AssemblyDirectory);
                searchAssemblyDirectory = (searchPath.Value & DllImportSearchPath.AssemblyDirectory) != 0;
            }
            else
            {
                GetDllImportSearchPathFlags(assembly, out searchPathFlags, out searchAssemblyDirectory);
            }

            LoadLibErrorTracker errorTracker = default;
            IntPtr ret = LoadBySearch(assembly, searchAssemblyDirectory, searchPathFlags, ref errorTracker, libraryName);
            if (throwOnError && ret == IntPtr.Zero)
            {
                errorTracker.Throw(libraryName);
            }

            return ret;
        }

        internal static IntPtr LoadBySearch(Assembly callingAssembly, bool searchAssemblyDirectory, int dllImportSearchPathFlags, ref LoadLibErrorTracker errorTracker, string libraryName)
        {
            IntPtr ret;

            int loadWithAlteredPathFlags = 0;
            bool libNameIsRelativePath = !Path.IsPathRooted(libraryName);

            foreach (LibraryNameVariation libraryNameVariation in LibraryNameVariation.DetermineLibraryNameVariations(libraryName, libNameIsRelativePath))
            {
                string currLibNameVariation = libraryNameVariation.Prefix + libraryName + libraryNameVariation.Suffix;

                if (!libNameIsRelativePath)
                {
                    int flags = loadWithAlteredPathFlags;
                    if ((dllImportSearchPathFlags & (int)DllImportSearchPath.UseDllDirectoryForDependencies) != 0)
                    {
                        // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR is the only flag affecting absolute path. Don't OR the flags
                        // unconditionally as all absolute path P/Invokes could then lose LOAD_WITH_ALTERED_SEARCH_PATH.
                        flags |= dllImportSearchPathFlags;
                    }

                    ret = LoadLibraryHelper(currLibNameVariation, flags, ref errorTracker);
                    if (ret != IntPtr.Zero)
                    {
                        return ret;
                    }
                }
                else if ((callingAssembly != null) && searchAssemblyDirectory)
                {
                    // Try to load the module alongside the assembly where the PInvoke was declared.
                    // This only makes sense in dynamic scenarios (JIT/interpreter), so leaving this out for now.
                }

                ret = LoadLibraryHelper(currLibNameVariation, dllImportSearchPathFlags, ref errorTracker);
                if (ret != IntPtr.Zero)
                {
                    return ret;
                }
            }

            return IntPtr.Zero;
        }

        internal static void GetDllImportSearchPathFlags(Assembly callingAssembly, out int searchPathFlags, out bool searchAssemblyDirectory)
        {
            var searchPath = DllImportSearchPath.AssemblyDirectory;

            foreach (CustomAttributeData cad in callingAssembly.CustomAttributes)
            {
                if (cad.AttributeType == typeof(DefaultDllImportSearchPathsAttribute))
                {
                    searchPath = (DllImportSearchPath)cad.ConstructorArguments[0].Value!;
                }
            }

            searchPathFlags = (int)(searchPath & ~DllImportSearchPath.AssemblyDirectory);
            searchAssemblyDirectory = (searchPath & DllImportSearchPath.AssemblyDirectory) != 0;
        }

        private static IntPtr LoadFromPath(string libraryName, bool throwOnError)
        {
            LoadLibErrorTracker errorTracker = default;
            IntPtr ret = LoadLibraryHelper(libraryName, 0, ref errorTracker);
            if (throwOnError && ret == IntPtr.Zero)
            {
                errorTracker.Throw(libraryName);
            }

            return ret;
        }

        private static IntPtr LoadLibraryHelper(string libraryName, int flags, ref LoadLibErrorTracker errorTracker)
        {
            IntPtr ret = LoadLibraryEx(libraryName, IntPtr.Zero, flags);
            if (ret != IntPtr.Zero)
            {
                return ret;
            }

            int lastError = Marshal.GetLastWin32Error();
            if (lastError != LoadLibErrorTracker.ERROR_INVALID_PARAMETER)
            {
                errorTracker.TrackErrorCode(lastError);
            }

            return ret;
        }

        internal struct LoadLibErrorTracker
        {
            internal const int ERROR_INVALID_PARAMETER = 0x57;
            internal const int ERROR_MOD_NOT_FOUND = 126;
            internal const int ERROR_BAD_EXE_FORMAT = 193;

            private int _errorCode;

            public void Throw(string libraryName)
            {
                if (_errorCode == ERROR_BAD_EXE_FORMAT)
                {
                    throw new BadImageFormatException();
                }

                throw new DllNotFoundException($"Unable to native library '{libraryName}' or one of its dependencies.");
            }

            public void TrackErrorCode(int errorCode)
            {
                _errorCode = errorCode;
            }

        }

        internal partial struct LibraryNameVariation
        {
            private const string LibraryNameSuffix = ".dll";

            public string Prefix;
            public string Suffix;

            public LibraryNameVariation(string prefix, string suffix)
            {
                Prefix = prefix;
                Suffix = suffix;
            }

            internal static IEnumerable<LibraryNameVariation> DetermineLibraryNameVariations(string libName, bool isRelativePath, bool forOSLoader = false)
            {
                // This is a copy of the logic in DetermineLibNameVariations in dllimport.cpp in CoreCLR

                yield return new LibraryNameVariation(string.Empty, string.Empty);

                // Follow LoadLibrary rules if forOSLoader is true
                if (isRelativePath &&
                    (!forOSLoader || libName.Contains('.') && !libName.EndsWith('.')) &&
                    !libName.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) &&
                    !libName.EndsWith(".exe", StringComparison.OrdinalIgnoreCase))
                {
                    yield return new LibraryNameVariation(string.Empty, LibraryNameSuffix);
                }
            }
        }
    }
}

namespace System.Diagnostics.CodeAnalysis
{
    [AttributeUsage(AttributeTargets.Parameter, Inherited = false)]
    internal sealed class MaybeNullWhenAttribute : Attribute
    {
        /// <summary>Initializes the attribute with the specified return value condition.</summary>
        /// <param name="returnValue">
        /// The return value condition. If the method returns this value, the associated parameter may be null.
        /// </param>
        public MaybeNullWhenAttribute(bool returnValue) => ReturnValue = returnValue;

        /// <summary>Gets the return value condition.</summary>
        public bool ReturnValue { get; }
    }

    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field | AttributeTargets.Parameter, Inherited = false)]
    internal sealed class DisallowNullAttribute : Attribute
    {
        public DisallowNullAttribute() { }
    }
}

namespace System.Linq
{
    internal static partial class NSEnumerable
    {
        public static IEnumerable<TResult> Zip<TFirst, TSecond, TResult>(this IEnumerable<TFirst> first, IEnumerable<TSecond> second, Func<TFirst, TSecond, TResult> resultSelector)
        {
            if (first is null)
            {
                throw new ArgumentNullException(nameof(first));
            }

            if (second is null)
            {
                throw new ArgumentNullException(nameof(second));
            }

            if (resultSelector is null)
            {
                throw new ArgumentNullException(nameof(resultSelector));
            }

            return ZipIterator(first, second, resultSelector);
        }

        public static IEnumerable<(TFirst First, TSecond Second)> Zip<TFirst, TSecond>(this IEnumerable<TFirst> first, IEnumerable<TSecond> second)
        {
            if (first is null)
            {
                throw new ArgumentNullException(nameof(first));
            }

            if (second is null)
            {
                throw new ArgumentNullException(nameof(second));
            }

            return ZipIterator(first, second);
        }

        public static IEnumerable<(TFirst First, TSecond Second, TThird Third)> Zip<TFirst, TSecond, TThird>(this IEnumerable<TFirst> first, IEnumerable<TSecond> second, IEnumerable<TThird> third)
        {
            if (first is null)
            {
                throw new ArgumentNullException(nameof(first));
            }

            if (second is null)
            {
                throw new ArgumentNullException(nameof(second));
            }

            if (third is null)
            {
                throw new ArgumentNullException(nameof(third));
            }

            return ZipIterator(first, second, third);
        }

        private static IEnumerable<(TFirst First, TSecond Second)> ZipIterator<TFirst, TSecond>(IEnumerable<TFirst> first, IEnumerable<TSecond> second)
        {
            using (IEnumerator<TFirst> e1 = first.GetEnumerator())
            using (IEnumerator<TSecond> e2 = second.GetEnumerator())
            {
                while (e1.MoveNext() && e2.MoveNext())
                {
                    yield return (e1.Current, e2.Current);
                }
            }
        }

        private static IEnumerable<TResult> ZipIterator<TFirst, TSecond, TResult>(IEnumerable<TFirst> first, IEnumerable<TSecond> second, Func<TFirst, TSecond, TResult> resultSelector)
        {
            using (IEnumerator<TFirst> e1 = first.GetEnumerator())
            using (IEnumerator<TSecond> e2 = second.GetEnumerator())
            {
                while (e1.MoveNext() && e2.MoveNext())
                {
                    yield return resultSelector(e1.Current, e2.Current);
                }
            }
        }

        private static IEnumerable<(TFirst First, TSecond Second, TThird Third)> ZipIterator<TFirst, TSecond, TThird>(IEnumerable<TFirst> first, IEnumerable<TSecond> second, IEnumerable<TThird> third)
        {
            using (IEnumerator<TFirst> e1 = first.GetEnumerator())
            using (IEnumerator<TSecond> e2 = second.GetEnumerator())
            using (IEnumerator<TThird> e3 = third.GetEnumerator())
            {
                while (e1.MoveNext() && e2.MoveNext() && e3.MoveNext())
                {
                    yield return (e1.Current, e2.Current, e3.Current);
                }
            }
        }
    }
}

namespace System.IO
{
    internal class NSPath
    {
        internal const char DirectorySeparatorChar = '\\'; // Windows implementation
        internal const char AltDirectorySeparatorChar = '/';
        internal const string DirectorySeparatorCharAsString = "\\";

        public static string Join(string path1, string path2)
        {
            if (path1 is null || path1.Length == 0)
                return path2;
            if (path2 is null || path2.Length == 0)
                return path1;

            bool hasSeparator = IsDirectorySeparator(path1[path1.Length - 1]) || IsDirectorySeparator(path2[0]);
            return hasSeparator ? string.Concat(path1, path2) : string.Concat(path1, DirectorySeparatorCharAsString, path2);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsDirectorySeparator(char c) => c == DirectorySeparatorChar || c == AltDirectorySeparatorChar;

        public static string Join(string path1, string path2, string path3)
        {
            if (path1 is null || path1.Length == 0)
                return Join(path2, path3);

            if (path2 is null || path2.Length == 0)
                return Join(path1, path3);

            if (path3 is null || path3.Length == 0)
                return Join(path1, path2);

            bool firstHasSeparator = IsDirectorySeparator(path1[path1.Length - 1]) || IsDirectorySeparator(path2[0]);
            bool secondHasSeparator = IsDirectorySeparator(path2[path2.Length - 1]) || IsDirectorySeparator(path3[0]);
            return path1 + (firstHasSeparator ? "" : DirectorySeparatorCharAsString) + path2 + (secondHasSeparator ? "" : DirectorySeparatorCharAsString) + path3;
        }

        public static string Join(string path1, string path2, string path3, string path4)
        {
            if (path1 is null || path1.Length == 0)
                return Join(path2, path3, path4);

            if (path2 is null || path2.Length == 0)
                return Join(path1, path3, path4);

            if (path3 is null || path3.Length == 0)
                return Join(path1, path2, path4);

            if (path4 is null || path4.Length == 0)
                return Join(path1, path2, path3);

            bool firstHasSeparator = IsDirectorySeparator(path1[path1.Length - 1]) || IsDirectorySeparator(path2[0]);
            bool secondHasSeparator = IsDirectorySeparator(path2[path2.Length - 1]) || IsDirectorySeparator(path3[0]);
            bool thirdHasSeparator = IsDirectorySeparator(path3[path3.Length - 1]) || IsDirectorySeparator(path4[0]);

            return path1 + (firstHasSeparator  ? "" : DirectorySeparatorCharAsString) +
                path2 + (secondHasSeparator ? "" : DirectorySeparatorCharAsString) +
                path3 + (thirdHasSeparator  ? "" : DirectorySeparatorCharAsString) +
                path4;
        }
    }
}