// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/a4f53308b2d0f1aa9191686e326f45c26053f686/torchvision/ops/utils.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE


using System;
using System.Collections.Generic;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            internal static Tensor _upcast(Tensor t)
            {
                switch (t.dtype) {
                case ScalarType.BFloat16:
                case ScalarType.Float16:
                    return t.@float();
                case ScalarType.Bool:
                case ScalarType.Byte:
                case ScalarType.Int8:
                case ScalarType.Int16:
                    return t.@int();
                }
                return t.alias();
            }

            internal static Tensor _upcast_non_float(Tensor t) => (!t.is_floating_point() ? t.@float() : t.alias());

            internal static (Tensor, Tensor, Tensor, Tensor) unwrap4(Tensor[] t)
            {
                if (t.Length != 4) throw new ArgumentException("Not the right length");
                return (t[0], t[1], t[2], t[3]);
            }

            internal static (Tensor, Tensor) _loss_inter_union(Tensor boxes1, Tensor boxes2)
            {
                var (x1, y1, x2, y2) = unwrap4(boxes1.unbind(dimension: -1));
                var (x1g, y1g, x2g, y2g) = unwrap4(boxes2.unbind(dimension: -1));

                // Intersection keypoints
                var xkis1 = max(x1, x1g);
                var ykis1 = max(y1, y1g);
                var xkis2 = min(x2, x2g);
                var ykis2 = min(y2, y2g);

                var intsctk = zeros_like(x1);
                var mask = (ykis2 > ykis1) & (xkis2 > xkis1);
                intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask]);
                var unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk;

                return (intsctk, unionk);
            }


            internal static (Tensor, Tensor) _diou_iou_loss(Tensor boxes1, Tensor boxes2, double eps = 1e-7)
            {
                var (intsct, union) = _loss_inter_union(boxes1, boxes2);
                var iou = intsct / (union + eps);
                // smallest enclosing box
                var (x1, y1, x2, y2) = unwrap4(boxes1.unbind(dimension: -1));
                var (x1g, y1g, x2g, y2g) = unwrap4(boxes2.unbind(dimension: -1));

                var xc1 = min(x1, x1g);
                var yc1 = min(y1, y1g);
                var xc2 = max(x2, x2g);
                var yc2 = max(y2, y2g);

                var diagonal_distance_squared = (xc2 - xc1).pow(2) + (yc2 - yc1).pow(2) + eps;
                // centers of boxes
                var x_p = (x2 + x1) / 2;
                var y_p = (y2 + y1) / 2;
                var x_g = (x1g + x2g) / 2;
                var y_g = (y1g + y2g) / 2;

                // The distance between boxes' centers squared.
                var centers_distance_squared = (x_p - x_g).pow(2) + (y_p - y_g).pow(2);
                // The distance IoU is the IoU penalized by a normalized
                // distance between boxes' centers squared.
                var loss = 1 - iou + (centers_distance_squared / diagonal_distance_squared);
                return (loss, iou);
            }

            internal static Tensor _cat(IList<Tensor> tensors, int dim = 0)
            {
                if (tensors.Count == 1)
                    return tensors[0];
                return cat(tensors, dim);
            }

            internal static Tensor convert_boxes_to_roi_format(IList<Tensor> boxes)
            {
                var concat_boxes = _cat(boxes);
                var temp = new List<Tensor>();
                var idx_1 = TensorIndex.Ellipsis;
                var idx_2 = TensorIndex.Slice(stop: 1);
                for (var i = 0; i < boxes.Count; i++) {
                    var b = boxes[i];
                    temp.Add(full_like(b[idx_1, idx_2], i));
                }
                var ids = _cat(temp, dim: 0);
                var rois = cat(new Tensor[] { ids, concat_boxes }, dim: 1);
                return rois;
            }

            internal static void check_roi_boxes_shape(Tensor boxes)
            {
                if (boxes.size(1) != 5)
                    throw new ArgumentException("The boxes tensor shape is not correct as Tensor[K, 5]");
            }

            internal static void check_roi_boxes_shape(IList<Tensor> boxes)
            {
                foreach (var _tensor in boxes) {
                    if (_tensor.size(1) != 4)
                        throw new ArgumentException("The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]");
                }
            }

            internal static Tensor _box_cxcywh_to_xyxy(Tensor boxes)
            {
                var (cx, cy, w, h) = unwrap4(boxes.unbind(dimension: -1));
                var x1 = cx - 0.5 * w;
                var y1 = cy - 0.5 * h;
                var x2 = cx + 0.5 * w;
                var y2 = cy + 0.5 * h;
                boxes = torch.stack(new[] { x1, y1, x2, y2 }, dim: -1);
                return boxes;
            }

            internal static Tensor _box_xyxy_to_cxcywh(Tensor boxes)
            {
                var (x1, y1, x2, y2) = unwrap4(boxes.unbind(dimension: -1));
                var cx = (x1 + x2) / 2;
                var cy = (y1 + y2) / 2;
                var w = x2 - x1;
                var h = y2 - y1;

                boxes = torch.stack(new[] { cx, cy, w, h }, dim: -1);
                return boxes;
            }

            internal static Tensor _box_xywh_to_xyxy(Tensor boxes)
            {
                var (x, y, w, h) = unwrap4(boxes.unbind(dimension: -1));
                boxes = torch.stack(new[] { x, y, x + w, y + h }, dim: -1);
                return boxes;
            }

            internal static Tensor _box_xyxy_to_xywh(Tensor boxes)
            {
                var (x1, y1, x2, y2) = unwrap4(boxes.unbind(dimension: -1));
                var w = x2 - x1;
                var h = y2 - y1;
                boxes = torch.stack(new[] { x1, y1, w, h }, dim: -1);
                return boxes;
            }
        }
    }
}
