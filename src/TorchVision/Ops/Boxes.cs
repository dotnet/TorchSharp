// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/a4f53308b2d0f1aa9191686e326f45c26053f686/torchvision/ops/boxes.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {

        public static partial class ops
        {
            /// <summary>
            /// Converts boxes from given in_fmt to out_fmt.
            /// Supported in_fmt and out_fmt are:
            ///
            ///     'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right. This is the format that torchvision utilities expect.
            ///     'xywh' : boxes are represented via corner, width and height, x1, y2 being top left, w, h being width and height.
            ///     'cxcywh' : boxes are represented via centre, width and height, cx, cy being center of box, w, h being width and height.
            /// </summary>
            /// <param name="boxes">Boxes which will be converted</param>
            /// <param name="in_fmt">Input format of given boxes.</param>
            /// <param name="out_fmt">Output format of given boxes.</param>
            /// <returns></returns>
            public static Tensor box_convert(Tensor boxes, BoxFormats in_fmt, BoxFormats out_fmt)
            {
                if (in_fmt == out_fmt) return boxes.clone();

                if (in_fmt != BoxFormats.xyxy && out_fmt != BoxFormats.xyxy) {
                    boxes = (in_fmt == BoxFormats.xywh) ? _box_xywh_to_xyxy(boxes) : _box_cxcywh_to_xyxy(boxes);
                    in_fmt = BoxFormats.xyxy;
                }

                if (in_fmt == BoxFormats.xyxy) {
                    boxes = (out_fmt == BoxFormats.xywh) ? _box_xyxy_to_xywh(boxes) : _box_xyxy_to_cxcywh(boxes);
                } else if (out_fmt == BoxFormats.xyxy) {
                    boxes = (in_fmt == BoxFormats.xywh) ? _box_xywh_to_xyxy(boxes) : _box_cxcywh_to_xyxy(boxes);
                }

                return boxes;
            }

            /// <summary>
            /// Computes the area of a set of bounding boxes, which are specified by their (x1, y1, x2, y2) coordinates.
            /// </summary>
            /// <param name="boxes">Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format.</param>
            /// <returns></returns>
            public static Tensor box_area(Tensor boxes)
            {
                boxes = _upcast(boxes);
                return (boxes[colon, 2] - boxes[colon, 0]) * (boxes[colon, 3] - boxes[colon, 1]);
            }

            /// <summary>
            /// Return intersection-over-union (Jaccard index) between two sets of boxes.
            /// </summary>
            /// <param name="boxes1">First set of boxes</param>
            /// <param name="boxes2">Second set of boxes</param>
            /// <returns>The NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2</returns>
            public static Tensor box_iou(Tensor boxes1, Tensor boxes2)
            {
                var inter = _box_inter_union(boxes1, boxes2, out var union);
                return inter / union;
            }

            /// <summary>
            /// Return generalized intersection-over-union (Jaccard index) between two sets of boxes.
            /// </summary>
            /// <param name="boxes1">First set of boxes</param>
            /// <param name="boxes2">Second set of boxes</param>
            /// <returns>The NxM matrix containing the pairwise generalized IoU values for every element in boxes1 and boxes2</returns>
            public static Tensor generalized_box_iou(Tensor boxes1, Tensor boxes2)
            {
                using var _ = NewDisposeScope();
                var inter = _box_inter_union(boxes1, boxes2, out var union);
                var iou = inter / union;

                var lti = torch.min(boxes1[colon, None, (null, 2)], boxes2[colon, (null, 2)]);
                var rbi = torch.max(boxes1[colon, None, (2, null)], boxes2[colon, (2, null)]);

                var whi = _upcast(rbi - lti).clamp(min: 0);  // [N,M,2]
                var areai = whi[colon, colon, 0] * whi[colon, colon, 1];

                return (iou - (areai - union) / areai).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Return complete intersection-over-union (Jaccard index) between two sets of boxes.
            /// </summary>
            /// <param name="boxes1">First set of boxes</param>
            /// <param name="boxes2">Second set of boxes</param>
            /// <param name="eps">Small number to prevent division by zero</param>
            /// <returns>The NxM matrix containing the complete distance IoU values for every element in boxes1 and boxes2</returns>
            public static Tensor complete_box_iou(Tensor boxes1, Tensor boxes2, double eps = 1e-7)
            {
                using var _ = NewDisposeScope();
                boxes1 = _upcast(boxes1);
                boxes2 = _upcast(boxes2);

                var diou = _box_diou_iou(boxes1, boxes2, out Tensor iou, eps);

                var w_pred = boxes1[colon, None, 2] - boxes1[colon, None, 0];
                var h_pred = boxes1[colon, None, 3] - boxes1[colon, None, 1];

                var w_gt = boxes2[colon, 2] - boxes2[colon, 0];
                var h_gt = boxes2[colon, 3] - boxes2[colon, 1];

                var v = (4 / (Math.Pow(Math.PI, 2))) * torch.pow(torch.atan(w_pred / h_pred) - torch.atan(w_gt / h_gt), 2);

                Tensor alpha;

                using (var ng = torch.no_grad()) {
                    alpha = v / (1 - iou + v + eps);
                }
                return (diou - alpha * v).MoveToOuterDisposeScope();
            }


            /// <summary>
            /// Return generalized intersection-over-union (Jaccard index) between two sets of boxes.
            /// </summary>
            /// <param name="boxes1">First set of boxes</param>
            /// <param name="boxes2">Second set of boxes</param>
            /// <param name="eps">Small number to prevent division by zero</param>
            /// <returns>The NxM matrix containing the pairwise distance IoU values for every element in boxes1 and boxes2</returns>
            public static Tensor distance_box_iou(Tensor boxes1, Tensor boxes2, double eps = 1e-7)
            {
                using var _ = NewDisposeScope();
                boxes1 = _upcast(boxes1);
                boxes2 = _upcast(boxes2);
                var diou = _box_diou_iou(boxes1, boxes2, out var _, eps: eps);
                return diou.MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Compute the bounding boxes around the provided masks.
            /// </summary>
            /// <param name="masks">masks to transform where N is the number of masks and (H, W) are the spatial dimensions.</param>
            /// <returns>A [N, 4] tensor containing bounding boxes. The boxes are in the (x1, y1, x2, y2) format</returns>
            /// <exception cref="ArgumentException">Raised if the input is not a three-dimensional tensor.</exception>
            public static Tensor masks_to_boxes(Tensor masks)
            {
                if (masks.ndim != 3) throw new ArgumentException("'masks' should have three dimensions: NxHxW");
                if (masks.numel() == 0) {
                    return torch.zeros(0, 4, device: masks.device, dtype: torch.float32);
                }

                var n = masks.shape[0];
                var bounding_boxes = torch.zeros(n, 4, device: masks.device, dtype: torch.float32);

                for (int i = 0;i < n; i++) {
                    var yx = torch.where(masks[i] != 0);
                    bounding_boxes[i, 0] = torch.min(yx[1]);
                    bounding_boxes[i, 1] = torch.min(yx[0]);
                    bounding_boxes[i, 2] = torch.max(yx[1]);
                    bounding_boxes[i, 3] = torch.max(yx[0]);
                }
                return bounding_boxes;
            }

            private static Tensor _box_inter_union(Tensor boxes1, Tensor boxes2, out Tensor union)
            {
                var area1 = box_area(boxes1);
                var area2 = box_area(boxes2);

                var lt = torch.max(boxes1[colon, None, (null, 2)], boxes2[colon, (null, 2)]);  // [N,M,2];
                var rb = torch.min(boxes1[colon, None, (2, null)], boxes2[colon, (2, null)]);  // [N,M,2];

                var wh = _upcast(rb - lt).clamp(min: 0); // [N,M,2];
                var inter = wh[colon, colon, 0] * wh[colon, colon, 1]; // [N,M];

                union = area1[colon, None] + area2 - inter;
                return inter;
            }

            private static Tensor _box_diou_iou(Tensor boxes1, Tensor boxes2, out Tensor iou, double eps = 1e-7)
            {
                iou = box_iou(boxes1, boxes2);
                var lti = torch.min(boxes1[colon, None, (null, 2)], boxes2[colon, (null, 2)]);
                var rbi = torch.max(boxes1[colon, None, (2, null)], boxes2[colon, (2, null)]);
                var whi = _upcast(rbi - lti).clamp(min: 0);  // [N,M,2];
                var diagonal_distance_squared = whi[colon, colon, 0].pow(2) + whi[colon, colon, 1].pow(2) + eps;
                // centers of boxes
                var x_p = (boxes1[colon, 0] + boxes1[colon, 2]) / 2;
                var y_p = (boxes1[colon, 1] + boxes1[colon, 3]) / 2;
                var x_g = (boxes2[colon, 0] + boxes2[colon, 2]) / 2;
                var y_g = (boxes2[colon, 1] + boxes2[colon, 3]) / 2;
                // The distance between boxes' centers squared.
                var centers_distance_squared = _upcast((x_p[colon, None] - x_g[None, colon])).pow(2) + _upcast((y_p[colon, None] - y_g[None, colon])).pow(2);
                // The distance IoU is the IoU penalized by a normalized
                // distance between boxes' centers squared.
                return iou - (centers_distance_squared / diagonal_distance_squared);
            }

            public enum BoxFormats
            {
                xyxy, xywh, cxcywh
            }

            private static TensorIndex colon = TensorIndex.Colon;
            private static TensorIndex None = TensorIndex.None;
        }
    }
}
