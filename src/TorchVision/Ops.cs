// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;
using System.Collections.Generic;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            public static Tensor sigmoid_focal_loss(Tensor inputs, Tensor targets, float alpha = 0.25f, float gamma = 2.0f, nn.Reduction reduction = nn.Reduction.None)
            {
                using (var _ = torch.NewDisposeScope()) {

                    var p = torch.sigmoid(inputs);
                    var ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction: reduction);
                    var p_t = p * targets + (1 - p) * (1 - targets);

                    var loss = ce_loss * (1 - p_t).pow(gamma);

                    if (alpha >= 0) {
                        var alpha_t = alpha * targets + (1 - alpha) * (1 - targets);
                        loss = alpha_t * loss;
                    }

                    if (reduction == nn.Reduction.Mean) {
                        loss = loss.mean();
                    } else if (reduction == nn.Reduction.Sum) {
                        loss = loss.sum();
                    }

                    return loss.MoveToOuterDisposeScope();
                }
            }

            /// <summary>
            /// Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU). NMS iteratively removes lower scoring boxes which have an IoU greater than iou_threshold with another(higher scoring) box.
            /// </summary>
            /// <param name="boxes">Boxes (Tensor[N, 4]) to perform NMS on. They are expected to be in (left, top, right, down) format.</param>
            /// <param name="scores">Scores (Tensor[N]) for each one of the boxes.</param>
            /// <param name="iou_threshold">Discards all overlapping boxes with IoU > iou_threshold.</param>
            /// <returns>The indices (Tensor) of the elements that have been kept by NMS, sorted in decreasing order of scores.</returns>
            public static Tensor nms(Tensor boxes, Tensor scores, double iou_threshold = 0.5)
            {
                using (var _ = torch.NewDisposeScope()) {
                    var x1 = boxes.select(1, 0);
                    var y1 = boxes.select(1, 1);
                    var x2 = boxes.select(1, 2);
                    var y2 = boxes.select(1, 3);
                    var areas = (x2 - x1) * (y2 - y1);
                    var (_, order) = scores.sort(0, descending: true);

                    var keep = new List<long>();
                    while (order.numel() > 0) {
                        long i;
                        if (order.numel() == 1) {
                            i = order.cpu().item<long>();
                            keep.Add(i);
                            break;
                        } else {
                            i = order[0].cpu().item<long>();
                            keep.Add(i);
                        }

                        var indices = torch.arange(1, order.shape[0], dtype: ScalarType.Int64);
                        order = order[indices];
                        var xx1 = x1[order].clamp(min: x1[i]);
                        var yy1 = y1[order].clamp(min: y1[i]);
                        var xx2 = x2[order].clamp(max: x2[i]);
                        var yy2 = y2[order].clamp(max: y2[i]);
                        var inter = (xx2 - xx1).clamp(min: 0) * (yy2 - yy1).clamp(min: 0);

                        var iou = inter / (areas[i] + areas[order] - inter);
                        var idx = (iou <= iou_threshold).nonzero().squeeze();
                        if (idx.numel() == 0) {
                            break;
                        }

                        order = order[idx];
                    }

                    var ids = torch.from_array(keep.ToArray()).to_type(ScalarType.Int64).to(device: boxes.device);
                    return ids.MoveToOuterDisposeScope();
                }
            }
        }
    }
}
