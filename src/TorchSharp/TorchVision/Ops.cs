// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    public static class ops
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
    }
}
