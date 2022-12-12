// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/ea0be26b88778b1033d4a176be68bcdd008ff934/torchvision/ops/boxes.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using static TorchSharp.torch;
using TorchSharp.Utils;
using static TorchSharp.torchvision;
using System.Drawing;
using TorchSharp.Modules;
using System.Xml.Linq;
using static TorchSharp.torchvision.models;
using System.Reflection;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Collections.Generic;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            /// <summary>
            /// Performs non-maximum suppression(NMS) on the boxes according
            /// to their intersection-over-union(IoU).
            /// NMS iteratively removes lower scoring boxes which have an
            /// IoU greater than iou_threshold with another(higher scoring)
            /// box.
            /// If multiple boxes have the exact same score and satisfy the IoU
            /// criterion with respect to a reference box, the selected box is
            /// not guaranteed to be the same between CPU and GPU.This is similar
            /// to the behavior of argsort in PyTorch when repeated values are present.
            /// </summary>
            /// <param name="boxes">boxes to perform NMS on. They are expected to be in ``(x1, y1, x2, y2)`` format with ``0 &lt;= x1&lt;x2`` and``0 &lt;= y1&lt;y2``.</param>
            /// <param name="scores">scores (Tensor[N]): scores for each one of the boxes</param>
            /// <param name="iou_threshold">discards all overlapping boxes with IoU > iou_threshold</param>
            /// <returns>int64 tensor with the indices of the elements that have been kept by NMS, sorted in decreasing order of scores</returns>
            public static Tensor nms(Tensor boxes, Tensor scores, double iou_threshold)
            {
                var res = LibTorchSharp.THSVision_nms(boxes.Handle, scores.Handle, iou_threshold);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Performs non-maximum suppression in a batched fashion.
            /// Each index value correspond to a category, and NMS
            /// will not be applied between elements of different categories.
            /// </summary>
            /// <param name="boxes">(Tensor[N, 4]): boxes where NMS will be performed. They
            ///            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 &lt;= x1 &lt; x2`` and
            ///            ``0 &lt;= y1 &lt; y2``.</param>
            /// <param name="scores">(Tensor[N]): scores for each one of the boxes</param>
            /// <param name="idxs">(Tensor[N]): indices of the categories for each one of the boxes.</param>
            /// <param name="iou_threshold">discards all overlapping boxes with IoU > iou_threshold</param>
            /// <returns>Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
            ///        in decreasing order of scores</returns>
            public static Tensor batched_nms(Tensor boxes, Tensor scores, Tensor idxs, double iou_threshold)
            {
                //    # Benchmarks that drove the following thresholds are at
                //    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339
                if (boxes.numel() > (boxes.device == torch.CPU ? 4000 : 20000))
                    return _batched_nms_vanilla(boxes, scores, idxs, iou_threshold);
                else
                    return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold);
            }

            private static Tensor _batched_nms_coordinate_trick(Tensor boxes, Tensor scores, Tensor idxs, double iou_threshold)
            {
                //    # strategy: in order to perform NMS independently per class,
                //    # we add an offset to all the boxes. The offset is dependent
                //    # only on the class idx, and is large enough so that boxes
                //    # from different classes do not overlap
                if (boxes.numel() == 0)
                    return torch.empty(new long[] { 0 }, dtype: torch.int64, device: boxes.device);
                var max_coordinate = boxes.max();
                var offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes));
                var boxes_for_nms = boxes + offsets[TensorIndex.Colon, TensorIndex.None];
                var keep = nms(boxes_for_nms, scores, iou_threshold);
                return keep;
            }

            private static Tensor _batched_nms_vanilla(Tensor boxes, Tensor scores, Tensor idxs, double iou_threshold)
            {
                //    # Based on Detectron2 implementation, just manually call nms() on each class independently
                var keep_mask = torch.zeros_like(scores, dtype: torch.@bool);
                var unique = torch.unique(idxs);
                for (int i = 0; i < unique.NumberOfElements; i++) {
                    var class_id = unique[i];
                    var curr_indices = torch.where(idxs == class_id)[0];
                    var curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold);
                    keep_mask[curr_indices[curr_keep_indices]] = true;
                }
                var keep_indices = torch.where(keep_mask)[0];
                return keep_indices[scores[keep_indices].sort(descending: true).Indices];
            }

            /// <summary>
            /// Remove boxes which contains at least one side smaller than min_size.
            /// </summary>
            /// <param name="boxes"></param>
            /// <param name="min_size"></param>
            /// <returns></returns>
            public static Tensor remove_small_boxes(Tensor boxes, double min_size)
            {
                var ws = boxes[TensorIndex.Colon, 2] - boxes[TensorIndex.Colon, 0];
                var hs = boxes[TensorIndex.Colon, 3] - boxes[TensorIndex.Colon, 1];
                var keep = (ws >= min_size) & (hs >= min_size);
                keep = torch.where(keep)[0];
                return keep;
            }

            /// <summary>
            /// Clip boxes so that they lie inside an image of size `size`.
            /// </summary>
            /// <param name="boxes"></param>
            /// <param name="size"></param>
            /// <returns></returns>
            public static Tensor clip_boxes_to_image(Tensor boxes, long[] size)
            {
                var dim = boxes.dim();
                var boxes_x = boxes[TensorIndex.Ellipsis, TensorIndex.Slice(start: 0, step: 2)];
                var boxes_y = boxes[TensorIndex.Ellipsis, TensorIndex.Slice(start: 1, step: 2)];
                var height = size[0];
                var width = size[1];

                boxes_x = boxes_x.clamp(min: 0, max: width);
                boxes_y = boxes_y.clamp(min: 0, max: height);

                var clipped_boxes = torch.stack(new Tensor[] { boxes_x, boxes_y }, dim: dim);
                return clipped_boxes.reshape(boxes.shape);
            }

            /// <summary>
            /// Converts boxes from given in_fmt to out_fmt.
            ///    Supported in_fmt and out_fmt are:
            ///    'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
            ///    This is the format that torchvision utilities expect.
            ///    'xywh' : boxes are represented via corner, width and height, x1, y2 being top left, w, h being width and height.
            ///    'cxcywh' : boxes are represented via centre, width and height, cx, cy being center of box, w, h
            ///    being width and height.
            /// </summary>
            /// <param name="boxes"></param>
            /// <param name="in_fmt"></param>
            /// <param name="out_fmt"></param>
            /// <returns></returns>
            public static Tensor box_convert(Tensor boxes, string in_fmt, string out_fmt)
            {
                var allowed_fmts = new List<string> { "xyxy", "xywh", "cxcywh" };
                if (!allowed_fmts.Contains(in_fmt) || !allowed_fmts.Contains(out_fmt))
                    throw new ArgumentException("Unsupported Bounding Box Conversions for given in_fmt and out_fmt");

                if (in_fmt == out_fmt)
                    return boxes.clone();

                if (in_fmt != "xyxy" && out_fmt != "xyxy") {
                    //# convert to xyxy and change in_fmt xyxy
                    if (in_fmt == "xywh")
                        boxes = _box_xywh_to_xyxy(boxes);
                    else if (in_fmt == "cxcywh")
                        boxes = _box_cxcywh_to_xyxy(boxes);
                    in_fmt = "xyxy";
                }

                if (in_fmt == "xyxy") {
                    if (out_fmt == "xywh")
                        boxes = _box_xyxy_to_xywh(boxes);
                    else if (out_fmt == "cxcywh")
                        boxes = _box_xyxy_to_cxcywh(boxes);
                } else if (out_fmt == "xyxy") {
                    if (in_fmt == "xywh")
                        boxes = _box_xywh_to_xyxy(boxes);
                    else if (in_fmt == "cxcywh")
                        boxes = _box_cxcywh_to_xyxy(boxes);
                }
                return boxes;
            }

            /// <summary>
            /// Computes the area of a set of bounding boxes, which are specified by their
            ///    (x1, y1, x2, y2) coordinates.
            /// </summary>
            /// <param name="boxes"></param>
            /// <returns></returns>
            public static Tensor box_area(Tensor boxes)
            {
                boxes = _upcast(boxes);
                return (boxes[TensorIndex.Colon, 2] - boxes[TensorIndex.Colon, 0]) * (boxes[TensorIndex.Colon, 3] - boxes[TensorIndex.Colon, 1]);
            }

            //# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
            //# with slight modifications
            private static (Tensor, Tensor) _box_inter_union(Tensor boxes1, Tensor boxes2)
            {
                var area1 = box_area(boxes1);
                var area2 = box_area(boxes2);

                var lt = torch.max(boxes1[TensorIndex.Colon, TensorIndex.None, TensorIndex.Slice(stop: 2)], boxes2[TensorIndex.Colon, TensorIndex.Slice(stop: 2)]);  // [N,M,2];
                var rb = torch.min(boxes1[TensorIndex.Colon, TensorIndex.None, TensorIndex.Slice(start: 2)], boxes2[TensorIndex.Colon, TensorIndex.Slice(start: 2)]);  // [N,M,2];

                var wh = _upcast(rb - lt).clamp(min: 0);  // [N,M,2];
                var inter = wh[TensorIndex.Colon, TensorIndex.Colon, 0] * wh[TensorIndex.Colon, TensorIndex.Colon, 1];  // [N,M];

                var union = area1[TensorIndex.Colon, TensorIndex.None] + area2 - inter;

                return (inter, union);
            }

            /// <summary>
            /// Return intersection-over-union (Jaccard index) between two sets of boxes.
            /// Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
            /// ``0 &lt;= x1&lt;x2`` and ``0 &lt;= y1&lt;y2``.
            /// </summary>
            /// <param name="boxes1">(Tensor[N, 4]): first set of boxes</param>
            /// <param name="boxes2">(Tensor[M, 4]): second set of boxes</param>
            /// <returns>Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2</returns>
            public static Tensor box_iou(Tensor boxes1, Tensor boxes2)
            {
                var (inter, union) = _box_inter_union(boxes1, boxes2);
                var iou = inter / union;
                return iou;
            }

            /// <summary>
            /// Implementation adapted from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
            /// Return generalized intersection-over-union (Jaccard index) between two sets of boxes.
            /// Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
            ///     ``0 &lt;= x1&lt;x2`` and ``0 &lt;= y1&lt;y2``.
            /// </summary>
            /// <param name="boxes1">(Tensor[N, 4]): first set of boxes</param>
            /// <param name="boxes2">(Tensor[M, 4]): second set of boxes</param>
            /// <returns>Tensor[N, M]: the NxM matrix containing the pairwise generalized IoU values
            ///     for every element in boxes1 and boxes2</returns>
            public static Tensor generalized_box_iou(Tensor boxes1, Tensor boxes2)
            {
                var (inter, union) = _box_inter_union(boxes1, boxes2);
                var iou = inter / union;

                var lti = torch.min(boxes1[TensorIndex.Colon, TensorIndex.None, TensorIndex.Slice(stop: 2)], boxes2[TensorIndex.Colon, TensorIndex.Slice(stop: 2)]);
                var rbi = torch.max(boxes1[TensorIndex.Colon, TensorIndex.None, TensorIndex.Slice(start: 2)], boxes2[TensorIndex.Colon, TensorIndex.Slice(start: 2)]);

                var whi = _upcast(rbi - lti).clamp(min: 0);//  # [N,M,2]
                var areai = whi[TensorIndex.Colon, TensorIndex.Colon, 0] * whi[TensorIndex.Colon, TensorIndex.Colon, 1];

                return iou - (areai - union) / areai;
            }

            /// <summary>
            /// Return complete intersection-over-union (Jaccard index) between two sets of boxes.
            /// Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
            ///     ``0 &lt;= x1&lt;x2`` and ``0 &lt;= y1&lt;y2``.
            /// </summary>
            /// <param name="boxes1">(Tensor[N, 4]): first set of boxes</param>
            /// <param name="boxes2">(Tensor[M, 4]): second set of boxes</param>
            /// <param name="eps">(float, optional): small number to prevent division by zero. Default: 1e-7</param>
            /// <returns>Tensor[N, M]: the NxM matrix containing the pairwise complete IoU values
            ///         for every element in boxes1 and boxes2</returns>
            public static Tensor complete_box_iou(Tensor boxes1, Tensor boxes2, float eps = 1e-7f)
            {
                boxes1 = _upcast(boxes1);
                boxes2 = _upcast(boxes2);

                var (diou, iou) = _box_diou_iou(boxes1, boxes2, eps);

                var w_pred = boxes1[TensorIndex.Colon, TensorIndex.None, 2] - boxes1[TensorIndex.Colon, TensorIndex.None, 0];
                var h_pred = boxes1[TensorIndex.Colon, TensorIndex.None, 3] - boxes1[TensorIndex.Colon, TensorIndex.None, 1];

                var w_gt = boxes2[TensorIndex.Colon, 2] - boxes2[TensorIndex.Colon, 0];
                var h_gt = boxes2[TensorIndex.Colon, 3] - boxes2[TensorIndex.Colon, 1];

                var v = (4 / (torch.pow(Math.PI, 2))) * torch.pow(torch.atan(w_pred / h_pred) - torch.atan(w_gt / h_gt), 2);
                Tensor alpha;
                using (torch.no_grad())
                    alpha = v / (1 - iou + v + eps);
                return diou - alpha * v;
            }


            /// <summary>
            /// Return distance intersection-over-union (Jaccard index) between two sets of boxes.
            /// Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
            ///     ``0 &lt;= x1&lt;x2`` and ``0 &lt;= y1&lt;y2``.
            /// </summary>
            /// <param name="boxes1">(Tensor[N, 4]): first set of boxes</param>
            /// <param name="boxes2">(Tensor[M, 4]): second set of boxes</param>
            /// <param name="eps">(float, optional): small number to prevent division by zero. Default: 1e-7</param>
            /// <returns>Tensor[N, M]: the NxM matrix containing the pairwise distance IoU values
            ///         for every element in boxes1 and boxes2</returns>
            public static Tensor distance_box_iou(Tensor boxes1, Tensor boxes2, float eps = 1e-7f)
            {
                boxes1 = _upcast(boxes1);
                boxes2 = _upcast(boxes2);
                var (diou, _) = _box_diou_iou(boxes1, boxes2, eps: eps);
                return diou;
            }

            private static (Tensor, Tensor) _box_diou_iou(Tensor boxes1, Tensor boxes2, float eps = 1e-7f)
            {
                var iou = box_iou(boxes1, boxes2);
                var lti = torch.min(boxes1[TensorIndex.Colon, TensorIndex.None, TensorIndex.Slice(stop: 2)], boxes2[TensorIndex.Colon, TensorIndex.Slice(stop: 2)]);
                var rbi = torch.max(boxes1[TensorIndex.Colon, TensorIndex.None, TensorIndex.Slice(start: 2)], boxes2[TensorIndex.Colon, TensorIndex.Slice(start: 2)]);
                var whi = _upcast(rbi - lti).clamp(min: 0);//  # [N,M,2]
                var diagonal_distance_squared = (torch.pow(whi[TensorIndex.Colon, TensorIndex.Colon, 0], 2)) + (torch.pow(whi[TensorIndex.Colon, TensorIndex.Colon, 1], 2)) + eps;
                //# centers of boxes
                var x_p = (boxes1[TensorIndex.Colon, 0] + boxes1[TensorIndex.Colon, 2]) / 2;
                var y_p = (boxes1[TensorIndex.Colon, 1] + boxes1[TensorIndex.Colon, 3]) / 2;
                var x_g = (boxes2[TensorIndex.Colon, 0] + boxes2[TensorIndex.Colon, 2]) / 2;
                var y_g = (boxes2[TensorIndex.Colon, 1] + boxes2[TensorIndex.Colon, 3]) / 2;
                //# The distance between boxes' centers squared.
                var centers_distance_squared = (torch.pow(_upcast((x_p[TensorIndex.Colon, TensorIndex.None] - x_g[TensorIndex.None, TensorIndex.Colon])), 2)) + (
                    torch.pow(_upcast((y_p[TensorIndex.Colon, TensorIndex.None] - y_g[TensorIndex.None, TensorIndex.Colon])), 2));
                //# The distance IoU is the IoU penalized by a normalized
                //# distance between boxes' centers squared.
                return (iou - (centers_distance_squared / diagonal_distance_squared), iou);
            }

            /// <summary>
            /// Compute the bounding boxes around the provided masks.
            /// Returns a[N, 4] tensor containing bounding boxes.The boxes are in ``(x1, y1, x2, y2)`` format with
            ///     ``0 &lt;= x1&lt;x2`` and ``0 &lt;= y1&lt;y2``.
            /// </summary>
            /// <param name="masks">(Tensor[N, H, W]): masks to transform where N is the number of masks
            /// and(H, W) are the spatial dimensions.</param>
            /// <returns>Tensor[N, 4]: bounding boxes</returns>
            public static Tensor masks_to_boxes(Tensor masks)
            {
                if (masks.numel() == 0)
                    return torch.zeros(new long[] { 0, 4 }, device: masks.device, dtype: torch.@float);

                var n = masks.shape[0];

                var bounding_boxes = torch.zeros(new long[] { n, 4 }, device: masks.device, dtype: torch.@float);

                for (int index = 0; index < masks.shape[0]; index++) {
                    var mask = masks[index];

                    var (y, x, _) = torch.where(mask != 0);

                    bounding_boxes[index, 0] = torch.min(x);
                    bounding_boxes[index, 1] = torch.min(y);
                    bounding_boxes[index, 2] = torch.max(x);
                    bounding_boxes[index, 3] = torch.max(y);
                }

                return bounding_boxes;
            }
        }
    }
}
