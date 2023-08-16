// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System.Collections.Generic;
using System.Threading.Tasks;
using static TorchSharp.torchvision.ops;
using Xunit;
using TorchSharp;

namespace TorchVision
{
    [Collection("Sequential")]
    public class TestTorchVisionOps
    {
        [Fact]
        public void NMS_OneBox()
        {
            using (var _ = torch.NewDisposeScope()) {
                var boxes = torch.from_array(new[,] {
                    { 0.0, 0.0, 80.0, 80.0 }
                });
                var scores = torch.from_array(new[] { 0.9 });

                var nms_boxes = nms(boxes, scores);
                Assert.Multiple(
                    () => Assert.Single(nms_boxes.shape),
                    () => Assert.Equal(1, nms_boxes.shape[0])
                );
            }
        }

        [Fact]
        public void NMS_MultipleBoxes()
        {
            using (var _ = torch.NewDisposeScope()) {
                // iou: 0.3913
                var boxes = torch.from_array(new[,] {
                    { 0.0, 0.0, 80.0, 80.0 },
                    { 20.0, 20.0, 100.0, 100.0 },
                });
                var scores = torch.from_array(new[] { 0.8, 0.9 });
                torch.Tensor nms_boxes = null;

                // Less than iou threshold.
                nms_boxes = nms(boxes, scores, 0.6);
                Assert.Multiple(
                    () => Assert.Single(nms_boxes.shape),
                    () => Assert.Equal(2, nms_boxes.shape[0]),
                    () => Assert.Equal(1, nms_boxes[0].cpu().item<long>()),
                    () => Assert.Equal(0, nms_boxes[1].cpu().item<long>())
                );

                // Larger than iou threshold.
                nms_boxes = nms(boxes, scores, 0.3);
                Assert.Multiple(
                    () => Assert.Single(nms_boxes.shape),
                    () => Assert.Equal(1, nms_boxes.shape[0]),
                    () => Assert.Equal(1, nms_boxes[0].cpu().item<long>())
                );
            }
        }

        [Fact(Skip="Introduces concurrency into unit test run")]
        public void NMS_Multithread()
        {
            // Multiple Thread.
            const int ThreadCount = 10;

            // Get multi-thread test result.
            var tasks = new List<Task>();
            for (var i = 0; i < ThreadCount; i++) {
                var task = Task.Run(() => {
                    NMS_OneBox();
                    NMS_MultipleBoxes();
                });

                tasks.Add(task);
            }

            Task.WaitAll(tasks.ToArray());
        }
    }
}
