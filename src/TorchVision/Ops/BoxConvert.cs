// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/3d60f498e71ba63b428edb184c9ac38fa3737fa6/torchvision/ops/_box_convert.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using static TorchSharp.torch;
using TorchSharp.Utils;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            /// <summary>
            /// Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
            /// (cx, cy) refers to center of bounding box
            /// (w, h) are width and height of bounding box
            /// </summary>
            /// <param name="boxes">boxes (Tensor[N, 4]): boxes in (cx, cy, w, h) format which will be converted.</param>
            /// <returns>boxes (Tensor(N, 4)): boxes in (x1, y1, x2, y2) format.</returns>
            internal static Tensor _box_cxcywh_to_xyxy(Tensor boxes)
            {
                //# We need to change all 4 of them so some temporary variable is needed.
                var (cx, cy, w, h, _) = boxes.unbind(-1);
                var x1 = cx - 0.5 * w;
                var y1 = cy - 0.5 * h;
                var x2 = cx + 0.5 * w;
                var y2 = cy + 0.5 * h;

                boxes = torch.stack(new Tensor[] { x1, y1, x2, y2 }, dim: -1);
                return boxes;
            }

            /// <summary>
            /// Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.
            /// (x1, y1) refer to top left of bounding box
            /// (x2, y2) refer to bottom right of bounding box
            /// </summary>
            /// <param name="boxes">boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format which will be converted.</param>
            /// <returns>boxes (Tensor(N, 4)): boxes in (cx, cy, w, h) format.</returns>
            internal static Tensor _box_xyxy_to_cxcywh(Tensor boxes)
            {
                var (x1, y1, x2, y2, _) = boxes.unbind(-1);
                var cx = (x1 + x2) / 2;
                var cy = (y1 + y2) / 2;
                var w = x2 - x1;
                var h = y2 - y1;

                boxes = torch.stack(new Tensor[] { cx, cy, w, h }, dim: -1);

                return boxes;
            }

            /// <summary>
            /// Converts bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.
            /// (x, y) refers to top left of bouding box.
            /// (w, h) refers to width and height of box.
            /// </summary>
            /// <param name="boxes">boxes (Tensor[N, 4]): boxes in (x, y, w, h) which will be converted.</param>
            /// <returns>boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format.</returns>
            internal static Tensor _box_xywh_to_xyxy(Tensor boxes)
            {
                var (x, y, w, h, _) = boxes.unbind(-1);
                boxes = torch.stack(new Tensor[] { x, y, x + w, y + h }, dim: -1);
                return boxes;
            }

            /// <summary>
            /// Converts bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.
            /// (x1, y1) refer to top left of bounding box
            /// (x2, y2) refer to bottom right of bounding box
            /// </summary>
            /// <param name="boxes">boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) which will be converted.</param>
            /// <returns>boxes (Tensor[N, 4]): boxes in (x, y, w, h) format.</returns>
            internal static Tensor _box_xyxy_to_xywh(Tensor boxes)
            {
                var (x1, y1, x2, y2, _) = boxes.unbind(-1);
                var w = x2 - x1;
                var h = y2 - y1;
                boxes = torch.stack(new Tensor[] { x1, y1, w, h }, dim: -1);
                return boxes;
            }
        }
    }
}
