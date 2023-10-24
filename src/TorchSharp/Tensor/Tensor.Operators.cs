// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable

namespace TorchSharp
{
    public static partial class torch
    {
        // this file contains operator overloads. Hopefully, they are self-explanatory.
        public partial class Tensor
        {
            public static Tensor operator +(Tensor left, Tensor right) => left.add(right);
            public static Tensor operator +(Tensor left, Scalar right) => left.add(right);
            public static Tensor operator +(Scalar left, Tensor right) => right.add(left);

            public static Tensor operator +(Tensor left, int right) => left.add(right);
            public static Tensor operator +(Tensor left, long right) => left.add(right);
            public static Tensor operator +(Tensor left, float right) => left.add(right);
            public static Tensor operator +(Tensor left, double right) => left.add(right);

            public static Tensor operator +(int left, Tensor right) => right.add(left);
            public static Tensor operator +(long left, Tensor right) => right.add(left);
            public static Tensor operator +(float left, Tensor right) => right.add(left);
            public static Tensor operator +(double left, Tensor right) => right.add(left);

            public static Tensor operator *(Tensor left, Tensor right) => left.mul(right);
            public static Tensor operator *(Tensor left, Scalar right) => left.mul(right);
            public static Tensor operator *(Scalar left, Tensor right) => right.mul(left);

            public static Tensor operator *(Tensor left, int right) => left.mul(right);
            public static Tensor operator *(Tensor left, long right) => left.mul(right);
            public static Tensor operator *(Tensor left, float right) => left.mul(right);
            public static Tensor operator *(Tensor left, double right) => left.mul(right);

            public static Tensor operator *(int left, Tensor right) => right.mul(left);
            public static Tensor operator *(long left, Tensor right) => right.mul(left);
            public static Tensor operator *(float left, Tensor right) => right.mul(left);
            public static Tensor operator *(double left, Tensor right) => right.mul(left);

            public static Tensor operator -(Tensor left, Tensor right) => left.sub(right);
            public static Tensor operator -(Tensor left, Scalar right) => left.sub(right);
            public static Tensor operator -(Scalar left, Tensor right) => right.negative().add(left);

            public static Tensor operator -(Tensor left, int right) => left.sub(right);
            public static Tensor operator -(Tensor left, long right) => left.sub(right);
            public static Tensor operator -(Tensor left, float right) => left.sub(right);
            public static Tensor operator -(Tensor left, double right) => left.sub(right);

            public static Tensor operator -(int left, Tensor right) => right.negative().add(left);
            public static Tensor operator -(long left, Tensor right) => right.negative().add(left);
            public static Tensor operator -(float left, Tensor right) => right.negative().add(left);
            public static Tensor operator -(double left, Tensor right) => right.negative().add(left);

            public static Tensor operator /(Tensor left, Tensor right) => left.div(right);
            public static Tensor operator /(Tensor left, Scalar right) => left.div(right);
            public static Tensor operator /(Scalar left, Tensor right) => right.reciprocal().mul(left);

            public static Tensor operator /(Tensor left, int right) => left.div(right);
            public static Tensor operator /(Tensor left, long right) => left.div(right);
            public static Tensor operator /(Tensor left, float right) => left.div(right);
            public static Tensor operator /(Tensor left, double right) => left.div(right);

            public static Tensor operator /(int left, Tensor right) => right.reciprocal().mul(left);
            public static Tensor operator /(long left, Tensor right) => right.reciprocal().mul(left);
            public static Tensor operator /(float left, Tensor right) => right.reciprocal().mul(left);
            public static Tensor operator /(double left, Tensor right) => right.reciprocal().mul(left);


            public static Tensor operator %(Tensor left, Tensor right) => left.remainder(right);
            public static Tensor operator %(Tensor left, Scalar right) => left.remainder(right);

            public static Tensor operator %(Tensor left, int right) => left.remainder(right);
            public static Tensor operator %(Tensor left, long right) => left.remainder(right);
            public static Tensor operator %(Tensor left, float right) => left.remainder(right);
            public static Tensor operator %(Tensor left, double right) => left.remainder(right);

            public static Tensor operator &(Tensor left, Tensor right) => left.bitwise_and(right);

            public static Tensor operator |(Tensor left, Tensor right) => left.bitwise_or(right);

            public static Tensor operator ^(Tensor left, Tensor right) => left.bitwise_xor(right);

            public static Tensor operator ~(Tensor left) => left.bitwise_not();
        }
    }
}