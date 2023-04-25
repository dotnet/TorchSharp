using System;
using System.IO;
using SkiaSharp;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class utils
        {
            public static partial class tensorboard
            {
                internal static partial class GifEncoder
                {
                    /// <summary>
                    /// Class AnimatedGifEncoder - Encodes a GIF file consisting of one or more frames.
                    ///
                    /// No copyright asserted on the source code of this class. May be used for any
                    /// purpose, however, refer to the Unisys LZW patent for restrictions on use of
                    /// the associated LZWEncoder class. Please forward any corrections to
                    /// kweiner@fmsware.com.
                    ///
                    /// @author Kevin Weiner, FM Software
                    /// @version 1.03 November 2003
                    ///
                    /// https://cs.android.com/android/platform/superproject/+/master:external/glide/third_party/gif_encoder/src/main/java/com/bumptech/glide/gifencoder/AnimatedGifEncoder.java
                    /// </summary>
                    internal class Encoder : IDisposable
                    {
                        protected int width; // image size
                        protected int height;
                        protected SKColor transparent = SKColor.Empty; // transparent color if given
                        protected int transIndex; // transparent index in color table
                        protected int repeat = -1; // no repeat
                        protected int delay = 0; // frame delay (hundredths)
                        protected bool started = false; // ready to output frames
                                                        //	protected BinaryWriter bw;
                        protected MemoryStream ms;
                        //		protected FileStream fs;

                        protected SKBitmap image; // current frame
                        protected byte[] pixels; // BGR byte array from frame
                        protected byte[] indexedPixels; // converted frame indexed to palette
                        protected int colorDepth; // number of bit planes
                        protected byte[] colorTab; // RGB palette
                        protected bool[] usedEntry = new bool[256]; // active palette entries
                        protected int palSize = 7; // color table size (bits-1)
                        protected int dispose = -1; // disposal code (-1 = use default)
                        protected bool closeStream = false; // close stream when finished
                        protected bool firstFrame = true;
                        protected bool sizeSet = false; // if false, get size from first frame
                        protected int sample = 10; // default sample interval for quantizer
                        private bool disposedValue;

                        /// <summary>
                        /// Sets the delay time between each frame, or changes it
                        /// for subsequent frames (applies to last frame added).
                        /// </summary>
                        /// <param name="ms"> delay time in milliseconds </param>
                        public void SetDelay(int ms)
                        {
                            delay = (int)Math.Round(ms / 10.0f);
                        }

                        /// <summary>
                        /// Sets the GIF frame disposal code for the last added frame
                        /// and any subsequent frames.  Default is 0 if no transparent
                        /// color has been set, otherwise 2.
                        /// </summary>
                        /// <param name="code"> disposal code. </param>
                        public void SetDispose(int code)
                        {
                            if (code >= 0) {
                                dispose = code;
                            }
                        }

                        /// <summary>
                        /// Sets the number of times the set of GIF frames
                        /// should be played.  Default is 1; 0 means play
                        /// indefinitely.  Must be invoked before the first
                        /// image is added.
                        /// </summary>
                        /// <param name="iter"> number of iterations. </param>
                        public void SetRepeat(int iter)
                        {
                            if (iter >= 0) {
                                repeat = iter;
                            }
                        }

                        /// <summary>
                        /// Sets the transparent color for the last added frame
                        /// and any subsequent frames.
                        /// Since all colors are subject to modification
                        /// in the quantization process, the color in the final
                        /// palette for each frame closest to the given color
                        /// becomes the transparent color for that frame.
                        /// May be set to null to indicate no transparent color.
                        /// </summary>
                        /// <param name="c"> Color to be treated as transparent on display. </param>
                        public void SetTransparent(SKColor c)
                        {
                            transparent = c;
                        }

                        /// <summary>
                        /// Adds next GIF frame.  The frame is not written immediately, but is
                        /// actually deferred until the next frame is received so that timing
                        /// data can be inserted.  Invoking <code>finish()</code> flushes all
                        /// frames.  If <code>setSize</code> was not invoked, the size of the
                        /// first image is used for all subsequent frames.
                        /// </summary>
                        /// <param name="im"> BufferedImage containing frame to write. </param>
                        /// <returns> true if successful. </returns>
                        public bool AddFrame(SKBitmap im)
                        {
                            if ((im == null) || !started) {
                                return false;
                            }
                            bool ok = true;
                            try {
                                if (!sizeSet) {
                                    // use first frame's size
                                    SetSize(im.Width, im.Height);
                                }
                                image = im;
                                GetImagePixels(); // convert to correct format if necessary
                                AnalyzePixels(); // build color table & map pixels
                                if (firstFrame) {
                                    WriteLSD(); // logical screen descriptior
                                    WritePalette(); // global color table
                                    if (repeat >= 0) {
                                        // use NS app extension to indicate reps
                                        WriteNetscapeExt();
                                    }
                                }
                                WriteGraphicCtrlExt(); // write graphic control extension
                                WriteImageDesc(); // image descriptor
                                if (!firstFrame) {
                                    WritePalette(); // local color table
                                }
                                WritePixels(); // encode and write pixel data
                                firstFrame = false;
                            } catch (IOException) {
                                ok = false;
                            }

                            return ok;
                        }

                        /// <summary>
                        /// Flushes any pending data and closes output file.
                        /// If writing to an OutputStream, the stream is not
                        /// closed.
                        /// </summary>
                        /// <returns></returns>
                        public bool Finish()
                        {
                            if (!started) return false;
                            bool ok = true;
                            started = false;
                            try {
                                ms.WriteByte(0x3b); // gif trailer
                                ms.Flush();
                            } catch (IOException) {
                                ok = false;
                            }

                            // reset for subsequent use
                            transIndex = 0;
                            //			fs = null;
                            image = null;
                            pixels = null;
                            indexedPixels = null;
                            colorTab = null;
                            closeStream = false;
                            firstFrame = true;

                            return ok;
                        }

                        /// <summary>
                        /// Sets frame rate in frames per second.  Equivalent to
                        /// <code>setDelay(1000/fps)</code>.
                        /// </summary>
                        /// <param name="fps"> fps float frame rate (frames per second) </param>
                        public void SetFrameRate(float fps)
                        {
                            if (fps != 0f) {
                                delay = (int)Math.Round(100f / fps);
                            }
                        }

                        /// <summary>
                        /// Sets the GIF frame size.  The default size is the
                        /// size of the first frame added if this method is
                        /// not invoked.
                        /// </summary>
                        /// <param name="w"> frame width </param>
                        /// <param name="h"> frame width </param>
                        public void SetSize(int w, int h)
                        {
                            if (started && !firstFrame) return;
                            width = w;
                            height = h;
                            if (width < 1) width = 320;
                            if (height < 1) height = 240;
                            sizeSet = true;
                        }

                        /// <summary>
                        /// Initiates GIF file creation on the given stream.  The stream
                        /// is not closed automatically.
                        /// </summary>
                        /// <param name="os"> OutputStream on which GIF images are written. </param>
                        /// <returns> false if initial write failed. </returns>
                        public bool Start(MemoryStream os)
                        {
                            if (os == null) return false;
                            bool ok = true;
                            closeStream = false;
                            ms = os;
                            try {
                                WriteString("GIF89a"); // header
                            } catch (IOException) {
                                ok = false;
                            }
                            return started = ok;
                        }

                        /// <summary>
                        /// Initiates writing of a GIF file to a memory stream.
                        /// </summary>
                        /// <returns></returns>
                        public bool Start()
                        {
                            bool ok;
                            try {
                                ok = Start(new MemoryStream(10 * 1024));
                                closeStream = true;
                            } catch (IOException) {
                                ok = false;
                            }
                            return started = ok;
                        }

                        /// <summary>
                        /// Initiates writing of a GIF file with the specified name.
                        /// </summary>
                        /// <param name="file"></param>
                        /// <returns></returns>
                        public bool Output(string file)
                        {
                            try {
                                var fs = new FileStream(file, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None);
                                fs.Write(ms.ToArray(), 0, (int)ms.Length);
                                fs.Close();
                            } catch (IOException) {
                                return false;
                            }
                            return true;
                        }

                        public MemoryStream Output()
                        {
                            return ms;
                        }

                        /// <summary>
                        /// Analyzes image colors and creates color map.
                        /// </summary>
                        protected void AnalyzePixels()
                        {
                            int len = pixels.Length;
                            int nPix = len / 3;
                            indexedPixels = new byte[nPix];
                            var nq = new NeuQuant(pixels, len, sample);
                            // initialize quantizer
                            colorTab = nq.Process(); // create reduced palette
                                                     // convert map from BGR to RGB
                                                     //			for (int i = 0; i < colorTab.Length; i += 3) 
                                                     //			{
                                                     //				byte temp = colorTab[i];
                                                     //				colorTab[i] = colorTab[i + 2];
                                                     //				colorTab[i + 2] = temp;
                                                     //				usedEntry[i / 3] = false;
                                                     //			}
                                                     // map image pixels to new palette
                            int k = 0;
                            for (int i = 0; i < nPix; i++) {
                                int index =
                                    nq.Map(pixels[k++] & 0xff,
                                    pixels[k++] & 0xff,
                                    pixels[k++] & 0xff);
                                usedEntry[index] = true;
                                indexedPixels[i] = (byte)index;
                            }
                            pixels = null;
                            colorDepth = 8;
                            palSize = 7;
                            // get closest match to transparent color if specified
                            if (transparent != SKColor.Empty) {
                                //transIndex = FindClosest(transparent);
                                transIndex = nq.Map(transparent.Blue, transparent.Green, transparent.Red);
                            }
                        }

                        /// <summary>
                        /// Returns index of palette color closest to c
                        /// </summary>
                        /// <param name="c"></param>
                        /// <returns></returns>
                        protected int FindClosest(SKColor c)
                        {
                            if (colorTab == null) return -1;
                            int r = c.Red;
                            int g = c.Green;
                            int b = c.Blue;
                            int minpos = 0;
                            int dmin = 256 * 256 * 256;
                            int len = colorTab.Length;
                            for (int i = 0; i < len;) {
                                int dr = r - (colorTab[i++] & 0xff);
                                int dg = g - (colorTab[i++] & 0xff);
                                int db = b - (colorTab[i] & 0xff);
                                int d = dr * dr + dg * dg + db * db;
                                int index = i / 3;
                                if (usedEntry[index] && (d < dmin)) {
                                    dmin = d;
                                    minpos = index;
                                }
                                i++;
                            }
                            return minpos;
                        }

                        /// <summary>
                        /// Extracts image pixels into byte array "pixels"
                        /// </summary>
                        protected void GetImagePixels()
                        {
                            pixels = new byte[3 * image.Width * image.Height];
                            int count = 0;

                            for (int th = 0; th < image.Height; th++) {
                                for (int tw = 0; tw < image.Width; tw++) {
                                    SKColor color = image.GetPixel(tw, th);
                                    pixels[count] = color.Red;
                                    count++;
                                    pixels[count] = color.Green;
                                    count++;
                                    pixels[count] = color.Blue;
                                    count++;
                                }
                            }

                            //		pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
                        }

                        /// <summary>
                        /// Writes Graphic Control Extension
                        /// </summary>
                        protected void WriteGraphicCtrlExt()
                        {
                            ms.WriteByte(0x21); // extension introducer
                            ms.WriteByte(0xf9); // GCE label
                            ms.WriteByte(4); // data block size
                            int transp, disp;
                            if (transparent == SKColor.Empty) {
                                transp = 0;
                                disp = 0; // dispose = no action
                            } else {
                                transp = 1;
                                disp = 2; // force clear if using transparent color
                            }
                            if (dispose >= 0) {
                                disp = dispose & 7; // user override
                            }
                            disp <<= 2;

                            // packed fields
                            ms.WriteByte(Convert.ToByte(0 | // 1:3 reserved
                                disp | // 4:6 disposal
                                0 | // 7   user input - 0 = none
                                transp)); // 8   transparency flag

                            WriteShort(delay); // delay x 1/100 sec
                            ms.WriteByte(Convert.ToByte(transIndex)); // transparent color index
                            ms.WriteByte(0); // block terminator
                        }

                        /// <summary>
                        /// Writes Image Descriptor
                        /// </summary>
                        protected void WriteImageDesc()
                        {
                            ms.WriteByte(0x2c); // image separator
                            WriteShort(0); // image position x,y = 0,0
                            WriteShort(0);
                            WriteShort(width); // image size
                            WriteShort(height);
                            // packed fields
                            if (firstFrame) {
                                // no LCT  - GCT is used for first (or only) frame
                                ms.WriteByte(0);
                            } else {
                                // specify normal LCT
                                ms.WriteByte(Convert.ToByte(0x80 | // 1 local color table  1=yes
                                    0 | // 2 interlace - 0=no
                                    0 | // 3 sorted - 0=no
                                    0 | // 4-5 reserved
                                    palSize)); // 6-8 size of color table
                            }
                        }

                        /// <summary>
                        /// Writes Logical Screen Descriptor
                        /// </summary>
                        protected void WriteLSD()
                        {
                            // logical screen size
                            WriteShort(width);
                            WriteShort(height);
                            // packed fields
                            ms.WriteByte(Convert.ToByte(0x80 | // 1   : global color table flag = 1 (gct used)
                                0x70 | // 2-4 : color resolution = 7
                                0x00 | // 5   : gct sort flag = 0
                                palSize)); // 6-8 : gct size

                            ms.WriteByte(0); // background color index
                            ms.WriteByte(0); // pixel aspect ratio - assume 1:1
                        }

                        /// <summary>
                        /// Writes Netscape application extension to define repeat count.
                        /// </summary>
                        protected void WriteNetscapeExt()
                        {
                            ms.WriteByte(0x21); // extension introducer
                            ms.WriteByte(0xff); // app extension label
                            ms.WriteByte(11); // block size
                            WriteString("NETSCAPE" + "2.0"); // app id + auth code
                            ms.WriteByte(3); // sub-block size
                            ms.WriteByte(1); // loop sub-block id
                            WriteShort(repeat); // loop count (extra iterations, 0=repeat forever)
                            ms.WriteByte(0); // block terminator
                        }

                        /// <summary>
                        /// Writes color table
                        /// </summary>
                        protected void WritePalette()
                        {
                            ms.Write(colorTab, 0, colorTab.Length);
                            int n = (3 * 256) - colorTab.Length;
                            for (int i = 0; i < n; i++) {
                                ms.WriteByte(0);
                            }
                        }

                        /// <summary>
                        /// Encodes and writes pixel data
                        /// </summary>
                        protected void WritePixels()
                        {
                            var encoder = new LZWEncoder(indexedPixels, colorDepth);
                            encoder.Encode(ms);
                        }

                        /// <summary>
                        /// Write 16-bit value to output stream, LSB first
                        /// </summary>
                        /// <param name="value"></param>
                        protected void WriteShort(int value)
                        {
                            ms.WriteByte(Convert.ToByte(value & 0xff));
                            ms.WriteByte(Convert.ToByte((value >> 8) & 0xff));
                        }

                        /// <summary>
                        /// Writes string to output stream
                        /// </summary>
                        /// <param name="s"></param>
                        protected void WriteString(string s)
                        {
                            char[] chars = s.ToCharArray();
                            for (int i = 0; i < chars.Length; i++) {
                                ms.WriteByte((byte)chars[i]);
                            }
                        }

                        protected virtual void Dispose(bool disposing)
                        {
                            if (!disposedValue) {
                                if (disposing) {
                                    ms.Dispose();
                                }

                                disposedValue = true;
                            }
                        }

                        ~Encoder()
                        {
                            Dispose(disposing: false);
                        }

                        public void Dispose()
                        {
                            Dispose(disposing: true);
                            GC.SuppressFinalize(this);
                        }
                    }
                }
            }
        }
    }
}
