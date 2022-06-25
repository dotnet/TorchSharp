// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
module TorchSharp.Examples.ImageTransforms

open System.IO

open TorchSharp

open type TorchSharp.torchvision.io
open type TorchSharp.torchvision.transforms


let run (argv : string[]) =
    DefaultImager <- new Utils.ImagerSharp()

    let img = read_image(argv[0])
    let shape = img.shape

    printfn "Image has %d color channels and dimensions %dx%d" (shape[0]) (shape[1]) (shape[2])

    let filename = Path.GetFileNameWithoutExtension(argv[0])
    let dir = Path.GetDirectoryName(argv[0])

    let transform = torchvision.transforms.Compose(HorizontalFlip(), Rotate(50f), CenterCrop(256))

    let transformed = transform.forward(img)

    let tfile = Path.Combine(dir, filename + "-transformed.png")

    write_image(transformed, tfile , torchvision.ImageFormat.Png)

    printfn "Wrote transformed image to %s" tfile

    let functional_transformed =
        functional.hflip(img)
        |> fun t -> functional.rotate(t, 50f)
        |> fun t -> functional.center_crop(t, 256)
        
    let ftfile = Path.Combine(dir, filename + "-functransformed.png")

    write_image(functional_transformed, ftfile, torchvision.ImageFormat.Png)

    printfn "Wrote functionaly transformed image to %s" ftfile
