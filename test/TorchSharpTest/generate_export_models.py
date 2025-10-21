#!/usr/bin/env python3
"""
Generate AOTInductor-compiled ExportedProgram models for TorchSharp testing.

This script creates .pt2 files using torch._inductor.aoti_compile_and_package(),
which compiles models with AOTInductor for inference-only execution in C++.

IMPORTANT: Models created with torch.export.save() alone cannot be loaded in LibTorch C++.
They must be compiled with aoti_compile_and_package() to create a loadable package.
"""

import torch
import torch.nn as nn
import torch._inductor
from pathlib import Path


class SimpleLinear(nn.Module):
    """Simple linear layer: 10 -> 5"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class LinearReLU(nn.Module):
    """Linear layer with ReLU: 10 -> 6"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 6)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class TwoInputs(nn.Module):
    """Model that takes two inputs and adds them"""
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


class TupleOutput(nn.Module):
    """Model that returns a tuple of (sum, difference)"""
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y, x - y


class ListOutput(nn.Module):
    """Model that returns a list of [sum, difference]"""
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return [x + y, x - y]


class Sequential(nn.Module):
    """Sequential model: 1000 -> 100 -> 50 -> 10"""
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        return self.seq(x)


def compile_and_save(model, example_inputs, output_path):
    """
    Export and compile a model with AOTInductor.

    Args:
        model: PyTorch module to export
        example_inputs: Tuple of example inputs for tracing
        output_path: Path where to save the .pt2 file
    """
    print(f"Compiling {output_path}...")

    # Set model to eval mode (inference only)
    model.eval()

    # Export the model
    with torch.no_grad():
        exported = torch.export.export(model, example_inputs)

    # Compile with AOTInductor and package
    # This creates a .pt2 file that can be loaded in LibTorch C++
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=str(output_path)
    )

    print(f"  ✓ Created {output_path}")


def main():
    print("Generating AOTInductor-compiled ExportedProgram models...\n")

    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # 1. Simple Linear (10 -> 5)
    model1 = SimpleLinear()
    compile_and_save(
        model1,
        (torch.ones(10),),
        script_dir / "simple_linear.export.pt2"
    )

    # 2. Linear + ReLU (10 -> 6)
    model2 = LinearReLU()
    compile_and_save(
        model2,
        (torch.ones(10),),
        script_dir / "linrelu.export.pt2"
    )

    # 3. Two Inputs (adds two tensors)
    model3 = TwoInputs()
    compile_and_save(
        model3,
        (torch.ones(10), torch.ones(10)),
        script_dir / "two_inputs.export.pt2"
    )

    # 4. Tuple Output (returns sum and difference)
    model4 = TupleOutput()
    compile_and_save(
        model4,
        (torch.rand(3, 4), torch.rand(3, 4)),
        script_dir / "tuple_out.export.pt2"
    )

    # 5. List Output (returns [sum, difference])
    model5 = ListOutput()
    compile_and_save(
        model5,
        (torch.rand(3, 4), torch.rand(3, 4)),
        script_dir / "list_out.export.pt2"
    )

    # 6. Sequential (1000 -> 100 -> 50 -> 10)
    model6 = Sequential()
    compile_and_save(
        model6,
        (torch.ones(1000),),
        script_dir / "sequential.export.pt2"
    )

    print("\n✓ All models compiled successfully!")
    print("\nThese models are now compatible with LibTorch C++ via")
    print("torch::inductor::AOTIModelPackageLoader for inference-only execution.")


if __name__ == "__main__":
    main()
