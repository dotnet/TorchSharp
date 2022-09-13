// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;
using System.Collections.Generic;

namespace TorchSharp.torchvision
{
    public static class ops
    {
        public static Tensor sigmoid_focal_loss(Tensor inputs, Tensor targets, float alpha = 0.25f, float gamma = 2.0f, nn.Reduction reduction = nn.Reduction.None)
        {
            using (var _ = torch.NewDisposeScope())
            {

                var p = torch.sigmoid(inputs);
                var ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction: reduction);
                var p_t = p * targets + (1 - p) * (1 - targets);

                var loss = ce_loss * (1 - p_t).pow(gamma);

                if (alpha >= 0)
                {
                    var alpha_t = alpha * targets + (1 - alpha) * (1 - targets);
                    loss = alpha_t * loss;
                }

                if (reduction == nn.Reduction.Mean)
                {
                    loss = loss.mean();
                }
                else if (reduction == nn.Reduction.Sum)
                {
                    loss = loss.sum();
                }

                return loss.MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU). NMS iteratively removes lower scoring boxes which have an IoU greater than iou_threshold with another(higher scoring) box.
        /// </summary>
        /// <param name="boxes">boxes (Tensor[N, 4]) to perform NMS on. They are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.</param>
        /// <param name="scores">scores (Tensor[N]) for each one of the boxes.</param>
        /// <param name="iou_threshold">discards all overlapping boxes with IoU > iou_threshold.</param>
        /// <returns>the indices (Tensor) of the elements that have been kept by NMS, sorted in decreasing order of scores.</returns>
        public static Tensor nme(Tensor boxes, Tensor scores, double iou_threshold = 0.5)
        {
            using (var nmsScope = torch.NewDisposeScope())
            {
                var x1 = boxes[.., 0];
                var y1 = boxes[.., 1];
                var x2 = boxes[.., 2];
                var y2 = boxes[.., 3];
                var areas = (x2 - x1) * (y2 - y1);
                var (_, _order) = scores.sort(0, descending: true);
                var order = _order[..];

                var keep = new List<long>();
                while (order.numel() > 0)
                {
                    long i;
                    if (order.numel() == 1)
                    {
                        i = order.item<long>();
                        keep.Add(i);
                        break;
                    }
                    else
                    {
                        i = order[0].item<long>();
                        keep.Add(i);
                    }

                    var xx1 = x1[order[1..]].clamp(min: x1[i]);
                    var yy1 = y1[order[1..]].clamp(min: y1[i]);
                    var xx2 = x2[order[1..]].clamp(max: x2[i]);
                    var yy2 = y2[order[1..]].clamp(max: y2[i]);
                    var inter = (xx2 - xx1).clamp(min: 0) * (yy2 - yy1).clamp(min: 0);

                    var iou = inter / (areas[i] + areas[order[1..]] - inter);
                    var idx = (iou <= iou_threshold).nonzero().squeeze();
                    if (idx.numel() == 0)
                    {
                        break;
                    }

                    order = order[idx + 1];
                }

                var ids = torch.from_array(keep.ToArray()).to_type(ScalarType.Int64).to(device: boxes.device);
                return ids.MoveToOuterDisposeScope();
            }
        }
    }
}
