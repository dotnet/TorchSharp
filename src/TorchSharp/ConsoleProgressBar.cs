// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Diagnostics;

namespace TorchSharp
{
    internal class ConsoleProgressBar : IProgressBar
    {
        private const long DisplayEveryMilliseconds = 100;

        private readonly int _displayWidth;
        private readonly Stopwatch _stopWatch;
        private bool _hidden;
        private long _value;
        private long? _maximum;

        internal ConsoleProgressBar(bool hidden)
        {
            this._hidden = hidden;
            try {
                // This fails when console is not available.
                this._displayWidth = Console.BufferWidth;
            } catch {
                this._displayWidth = 80;
            }
            this._stopWatch = new Stopwatch();
            this._stopWatch.Start();
        }

        public long Value {
            set {
                if (value > this._maximum) {
                    value = this._maximum.Value;
                }
                this._value = value;
                Display();
            }
            get {
                return _value;
            }
        }

        public long? Maximum {
            get {
                return _maximum;
            }
            set {
                if (value < this._value) {
                    this._value = value.Value;
                }
                this._maximum = value;
                Display();
            }
        }

        public void Dispose()
        {
            if (!_hidden) {
                Console.Error.WriteLine();
            }
        }

        private void Display()
        {
            if (!_hidden) {
                if (_value == 0 || _value == _maximum || this._stopWatch.ElapsedMilliseconds > DisplayEveryMilliseconds) {
                    this._stopWatch.Restart();
                    if (this.Maximum == null) {
                        Console.Error.Write("\r{0}", _value);
                    } else {
                        string left = string.Format("{0,3}%[", 100 * _value / _maximum);
                        string right = string.Format("] {0}/{1}", _value, _maximum);
                        int barContainerWidth = this._displayWidth - left.Length - right.Length - 1;
                        string center = string.Empty;
                        if (barContainerWidth > 0) {
                            int barWidth = (int)(barContainerWidth * _value / _maximum);
                            if (barWidth > 0) {
                                center = new string('=', barWidth);
                                if (this._displayWidth - barWidth > 0) {
                                    center += new string(' ', barContainerWidth - barWidth);
                                }
                            } else {
                                center = new string(' ', barContainerWidth);
                            }
                        }
                        Console.Error.Write("\r{0}{1}{2}", left, center, right);
                    }
                }
            }
        }
    }
}