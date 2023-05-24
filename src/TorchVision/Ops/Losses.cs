// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Drawing;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.torchvision;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            // Ported from https://github.com/pytorch/vision/blob/main/torchvision/ops/focal_loss.py

            /// <summary>
            /// Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
            /// </summary>
            /// <param name="inputs">A float tensor of arbitrary shape. The predictions for each example. </param>
            /// <param name="targets">A float tensor with the same shape as inputs.
            /// Stores the binary classification label for each element in inputs
            /// (0 for the negative class and 1 for the positive class).
            /// </param>
            /// <param name="alpha">Weighting factor in range (0,1) to balance positive vs negative examples or -1 for ignore.</param>
            /// <param name="gamma">Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.</param>
            /// <param name="reduction">The kind of reduction to apply to the result.</param>
            /// <returns></returns>
            public static Tensor sigmoid_focal_loss(
                Tensor inputs,
                Tensor targets,
                double alpha = 0.25, double gamma = 2.0,
                nn.Reduction reduction = nn.Reduction.None)
            {
                var p = torch.sigmoid(inputs);
                var ce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction: nn.Reduction.None);
                var p_t = p * targets + (1 - p) * (1 - targets);
                var loss = ce_loss * (1 - p_t).pow(gamma);

                if (alpha >= 0) {
                    var alpha_t = alpha * targets + (1 - alpha) * (1 - targets);
                    loss = alpha_t * loss;
                }

                switch (reduction) {
                case nn.Reduction.None:
                    break;
                case nn.Reduction.Mean:
                    loss = loss.mean();
                    break;
                case nn.Reduction.Sum:
                    loss = loss.sum();
                    break;
                }

                return loss;
            }

            /// <summary>
            /// Gradient-friendly IoU loss with an additional penalty that is non-zero when the
            /// boxes do not overlap.This loss function considers important geometrical
            /// factors such as overlap area, normalized central point distance and aspect ratio.
            /// This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.
            /// </summary>
            /// <param name="boxes1">(Tensor[N, 4] or Tensor[4]) first set of boxes</param>
            /// <param name="boxes2">(Tensor[N, 4] or Tensor[4]) second set of boxes</param>
            /// <param name="reduction">The kind of reduction to apply to the result.</param>
            /// <param name="eps">Small number to prevent division by zero.</param>
            public static Tensor complete_box_iou_loss(
                Tensor boxes1,
                Tensor boxes2,
                nn.Reduction reduction = nn.Reduction.None,
                double eps = 1e-7)
            {
                boxes1 = _upcast_non_float(boxes1);
                boxes2 = _upcast_non_float(boxes2);

                var (diou_loss, iou) = _diou_iou_loss(boxes1, boxes2);

                var xy = boxes1.unbind(dimension: -1);
                var xyg = boxes2.unbind(dimension: -1);

                // width and height of boxes
                var w_pred = xy[2] - xy[0];
                var h_pred = xy[3] - xy[1];
                var w_gt = xyg[2] - xyg[0];
                var h_gt = xyg[3] - xyg[1];

                var v = (4 / Math.Pow(Math.PI, 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2);

                Tensor alpha = null;

                using (var _ = torch.no_grad()) {
                    alpha = v / (1 - iou + v + eps);
                }

                var loss = diou_loss + alpha * v;

                switch (reduction) {
                case nn.Reduction.None:
                    break;
                case nn.Reduction.Mean:
                    loss = loss.mean();
                    break;
                case nn.Reduction.Sum:
                    loss = loss.sum();
                    break;
                }

                return loss;
            }

            /// <summary>
            /// Gradient-friendly IoU loss with an additional penalty that is non-zero when the
            /// distance between boxes' centers isn't zero.Indeed, for two exactly overlapping
            /// boxes, the distance IoU is the same as the IoU loss.
            /// This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.
            /// </summary>
            /// <param name="boxes1">(Tensor[N, 4] or Tensor[4]) first set of boxes</param>
            /// <param name="boxes2">(Tensor[N, 4] or Tensor[4]) second set of boxes</param>
            /// <param name="reduction">The kind of reduction to apply to the result.</param>
            /// <param name="eps">Small number to prevent division by zero.</param>
            public static Tensor distance_box_iou_loss(
                Tensor boxes1,
                Tensor boxes2,
                nn.Reduction reduction = nn.Reduction.None,
                double eps = 1e-7)
            {
                boxes1 = _upcast_non_float(boxes1);
                boxes2 = _upcast_non_float(boxes2);

                var (loss, _) = _diou_iou_loss(boxes1, boxes2);

                switch (reduction) {
                case nn.Reduction.None:
                    break;
                case nn.Reduction.Mean:
                    loss = loss.mean();
                    break;
                case nn.Reduction.Sum:
                    loss = loss.sum();
                    break;
                }

                return loss;
            }

            /// <summary>
            /// Gradient-friendly IoU loss with an additional penalty that is non-zero when the
            /// boxes do not overlap and scales with the size of their smallest enclosing box.
            /// This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.
            /// </summary>
            /// <param name="boxes1">(Tensor[N, 4] or Tensor[4]) first set of boxes</param>
            /// <param name="boxes2">(Tensor[N, 4] or Tensor[4]) second set of boxes</param>
            /// <param name="reduction">The kind of reduction to apply to the result.</param>
            /// <param name="eps">Small number to prevent division by zero.</param>
            public static Tensor generalized_box_iou_loss(
                Tensor boxes1,
                Tensor boxes2,
                nn.Reduction reduction = nn.Reduction.None,
                double eps = 1e-7)
            {
                boxes1 = _upcast_non_float(boxes1);
                boxes2 = _upcast_non_float(boxes2);

                var (intsctk, unionk) = _loss_inter_union(boxes1, boxes2);
                var iouk = intsctk / (unionk + eps);

                var (x1, y1, x2, y2) = unwrap4(boxes1.unbind(dimension: -1));
                var (x1g, y1g, x2g, y2g) = unwrap4(boxes2.unbind(dimension: -1));

                // smallest enclosing box

                var xc1 = torch.min(x1, x1g);
                var yc1 = torch.min(y1, y1g);
                var xc2 = torch.max(x2, x2g);
                var yc2 = torch.max(y2, y2g);

                var area_c = (xc2 - xc1) * (yc2 - yc1);
                var miouk = iouk - ((area_c - unionk) / (area_c + eps));

                var loss = 1 - miouk;

                switch (reduction) {
                case nn.Reduction.None:
                    break;
                case nn.Reduction.Mean:
                    loss = loss.mean();
                    break;
                case nn.Reduction.Sum:
                    loss = loss.sum();
                    break;
                }

                return loss;
            }
        }
    }
}
