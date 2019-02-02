//
// Internal bindings to support the NN bindings of Torch
//
//

using System;
using System.Runtime.InteropServices;
namespace AtenSharp {
	public partial class DoubleTensor {
		[DllImport ("caffe2")]
		extern static void AbsCriterion_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType target, DoubleTensor.HType output, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="target">tensor with target values</param>
		/// <param name="output">Returned value a one-element tensor with loss</param>
		/// <param name="reduction"></param>
		internal static void AbsCriterion_updateOutput(DoubleTensor input, DoubleTensor target, DoubleTensor output, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			AbsCriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void AbsCriterion_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType target, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="target">tensor with target values</param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput">Returned value gradient with regards to input</param>
		/// <param name="reduction"></param>
		internal static void AbsCriterion_updateGradInput(DoubleTensor input, DoubleTensor target, DoubleTensor gradOutput, DoubleTensor gradInput, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			AbsCriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void BCECriterion_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType target, DoubleTensor.HType output, long reduction, DoubleTensor.HType weights);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <param name="reduction"></param>
		/// <param name="weights"></param>
		internal static void BCECriterion_updateOutput(DoubleTensor input, DoubleTensor target, DoubleTensor output, long reduction, DoubleTensor weights)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weights == null)
				throw new ArgumentNullException (nameof (weights));

			BCECriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction, weights.handle);
		}


		[DllImport ("caffe2")]
		extern static void BCECriterion_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType target, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, long reduction, DoubleTensor.HType weights);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="reduction"></param>
		/// <param name="weights"></param>
		internal static void BCECriterion_updateGradInput(DoubleTensor input, DoubleTensor target, DoubleTensor gradOutput, DoubleTensor gradInput, long reduction, DoubleTensor weights)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weights == null)
				throw new ArgumentNullException (nameof (weights));

			BCECriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction, weights.handle);
		}


		[DllImport ("caffe2")]
		extern static void ClassNLLCriterion_updateOutput(IntPtr state, DoubleTensor.HType input, LongTensor.HType target, DoubleTensor.HType output, long reduction, DoubleTensor.HType weights, DoubleTensor.HType total_weight, long ignore_index);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor (1D/2D)</param>
		/// <param name="target">tensor containing indexes of target classes</param>
		/// <param name="output">Returned value a one-element tensor with loss</param>
		/// <param name="reduction"></param>
		/// <param name="weights">Optional, can be null. class weights</param>
		/// <param name="total_weight">A buffer.  The _updateGradInput and _accGradParameters methods should get the same buffers that were used in _updateOutput call.</param>
		/// <param name="ignore_index"></param>
		internal static void ClassNLLCriterion_updateOutput(DoubleTensor input, LongTensor target, DoubleTensor output, long reduction, DoubleTensor weights, DoubleTensor total_weight, long ignore_index)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (total_weight == null)
				throw new ArgumentNullException (nameof (total_weight));

			ClassNLLCriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction, weights == null ? null : weights.handle, total_weight.handle, ignore_index);
		}


		[DllImport ("caffe2")]
		extern static void ClassNLLCriterion_updateGradInput(IntPtr state, DoubleTensor.HType input, LongTensor.HType target, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, long reduction, DoubleTensor.HType weights, DoubleTensor.HType total_weight, long ignore_index);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor (1D/2D)</param>
		/// <param name="target">tensor containing indexes of target classes</param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput">Returned value gradient with regards to input</param>
		/// <param name="reduction"></param>
		/// <param name="weights">Optional, can be null. class weights</param>
		/// <param name="total_weight">A buffer.  The _updateGradInput and _accGradParameters methods should get the same buffers that were used in _updateOutput call.</param>
		/// <param name="ignore_index"></param>
		internal static void ClassNLLCriterion_updateGradInput(DoubleTensor input, LongTensor target, DoubleTensor gradOutput, DoubleTensor gradInput, long reduction, DoubleTensor weights, DoubleTensor total_weight, long ignore_index)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (total_weight == null)
				throw new ArgumentNullException (nameof (total_weight));

			ClassNLLCriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction, weights == null ? null : weights.handle, total_weight.handle, ignore_index);
		}


		[DllImport ("caffe2")]
		extern static void ELU_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, double alpha, double scale, double input_scale, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="output">Returned value ELU output</param>
		/// <param name="alpha">an ELU parameter (as in paper)</param>
		/// <param name="scale"></param>
		/// <param name="input_scale"></param>
		/// <param name="inplace"></param>
		internal static void ELU_updateOutput(DoubleTensor input, DoubleTensor output, double alpha, double scale, double input_scale, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			ELU_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, alpha, scale, input_scale, (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void ELU_updateGradInput(IntPtr state, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType output, double alpha, double scale, double input_scale);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput">gradient with regards to output</param>
		/// <param name="gradInput">Returned value gradient with regards to input</param>
		/// <param name="output">output from a forward pass</param>
		/// <param name="alpha">an ELU parameter (as in paper)</param>
		/// <param name="scale"></param>
		/// <param name="input_scale"></param>
		internal static void ELU_updateGradInput(DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor output, double alpha, double scale, double input_scale)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			ELU_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, output.handle, alpha, scale, input_scale);
		}


		[DllImport ("caffe2")]
		extern static void GatedLinear_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int dim);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="dim"></param>
		internal static void GatedLinear_updateOutput(DoubleTensor input, DoubleTensor output, int dim)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			GatedLinear_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, dim);
		}


		[DllImport ("caffe2")]
		extern static void GatedLinear_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int dim);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="dim"></param>
		internal static void GatedLinear_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, int dim)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			GatedLinear_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, dim);
		}


		[DllImport ("caffe2")]
		extern static void HardTanh_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, double min_val, double max_val, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="output">Returned value output tensor</param>
		/// <param name="min_val">lower threshold</param>
		/// <param name="max_val">upper threshold</param>
		/// <param name="inplace"></param>
		internal static void HardTanh_updateOutput(DoubleTensor input, DoubleTensor output, double min_val, double max_val, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			HardTanh_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, min_val, max_val, (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void HardTanh_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, double min_val, double max_val, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="gradOutput">gradient with regards to module's output</param>
		/// <param name="gradInput">Returned value gradient with regards to the input</param>
		/// <param name="min_val">lower threshold</param>
		/// <param name="max_val">upper threshold</param>
		/// <param name="inplace"></param>
		internal static void HardTanh_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, double min_val, double max_val, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			HardTanh_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, min_val, max_val, (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void Im2Col_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="kH"></param>
		/// <param name="kW"></param>
		/// <param name="dH"></param>
		/// <param name="dW"></param>
		/// <param name="padH"></param>
		/// <param name="padW"></param>
		/// <param name="sH"></param>
		/// <param name="sW"></param>
		internal static void Im2Col_updateOutput(DoubleTensor input, DoubleTensor output, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			Im2Col_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, kH, kW, dH, dW, padH, padW, sH, sW);
		}


		[DllImport ("caffe2")]
		extern static void Im2Col_updateGradInput(IntPtr state, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, long inputHeight, long inputWidth, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="inputHeight"></param>
		/// <param name="inputWidth"></param>
		/// <param name="kH"></param>
		/// <param name="kW"></param>
		/// <param name="dH"></param>
		/// <param name="dW"></param>
		/// <param name="padH"></param>
		/// <param name="padW"></param>
		/// <param name="sH"></param>
		/// <param name="sW"></param>
		internal static void Im2Col_updateGradInput(DoubleTensor gradOutput, DoubleTensor gradInput, long inputHeight, long inputWidth, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			Im2Col_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, inputHeight, inputWidth, kH, kW, dH, dW, padH, padW, sH, sW);
		}


		[DllImport ("caffe2")]
		extern static void Col2Im_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, long outputHeight, long outputWidth, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="outputHeight"></param>
		/// <param name="outputWidth"></param>
		/// <param name="kH"></param>
		/// <param name="kW"></param>
		/// <param name="dH"></param>
		/// <param name="dW"></param>
		/// <param name="padH"></param>
		/// <param name="padW"></param>
		/// <param name="sH"></param>
		/// <param name="sW"></param>
		internal static void Col2Im_updateOutput(DoubleTensor input, DoubleTensor output, long outputHeight, long outputWidth, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			Col2Im_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, outputHeight, outputWidth, kH, kW, dH, dW, padH, padW, sH, sW);
		}


		[DllImport ("caffe2")]
		extern static void Col2Im_updateGradInput(IntPtr state, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="kH"></param>
		/// <param name="kW"></param>
		/// <param name="dH"></param>
		/// <param name="dW"></param>
		/// <param name="padH"></param>
		/// <param name="padW"></param>
		/// <param name="sH"></param>
		/// <param name="sW"></param>
		internal static void Col2Im_updateGradInput(DoubleTensor gradOutput, DoubleTensor gradInput, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			Col2Im_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, kH, kW, dH, dW, padH, padW, sH, sW);
		}


		[DllImport ("caffe2")]
		extern static void LeakyReLU_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, double negval, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input">**[MODIFIED]** input tensor</param>
		/// <param name="output">Returned value output tensor</param>
		/// <param name="negval">negative part slope</param>
		/// <param name="inplace">if true, modifies the input tensor and sets the output tensor on it (no additional memory is allocated)</param>
		internal static void LeakyReLU_updateOutput(DoubleTensor input, DoubleTensor output, double negval, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			LeakyReLU_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, negval, (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void LeakyReLU_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, double negval, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="gradOutput">**[MODIFIED]** gradient with regards to module's output</param>
		/// <param name="gradInput">Returned value gradient with regards to the input</param>
		/// <param name="negval">negative part slope</param>
		/// <param name="inplace">if true, modifies gradOutput and sets gradInput onto it (no additional memory is allocated)</param>
		internal static void LeakyReLU_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, double negval, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			LeakyReLU_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, negval, (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void LogSigmoid_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, DoubleTensor.HType buffer);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="output">output tensor</param>
		/// <param name="buffer">A buffer.  The _updateGradInput and _accGradParameters methods should get the same buffers that were used in _updateOutput call.</param>
		internal static void LogSigmoid_updateOutput(DoubleTensor input, DoubleTensor output, DoubleTensor buffer)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (buffer == null)
				throw new ArgumentNullException (nameof (buffer));

			LogSigmoid_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, buffer.handle);
		}


		[DllImport ("caffe2")]
		extern static void LogSigmoid_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType buffer);
		/// <summary>
		/// </summary>
		/// <param name="input">input</param>
		/// <param name="gradOutput">gradient with regards to module's output</param>
		/// <param name="gradInput">Returned value gradient with regards to input</param>
		/// <param name="buffer">A buffer.  The _updateGradInput and _accGradParameters methods should get the same buffers that were used in _updateOutput call.</param>
		internal static void LogSigmoid_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor buffer)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (buffer == null)
				throw new ArgumentNullException (nameof (buffer));

			LogSigmoid_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, buffer.handle);
		}


		[DllImport ("caffe2")]
		extern static void SoftMarginCriterion_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType target, DoubleTensor.HType output, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <param name="reduction"></param>
		internal static void SoftMarginCriterion_updateOutput(DoubleTensor input, DoubleTensor target, DoubleTensor output, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SoftMarginCriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void SoftMarginCriterion_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType target, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="reduction"></param>
		internal static void SoftMarginCriterion_updateGradInput(DoubleTensor input, DoubleTensor target, DoubleTensor gradOutput, DoubleTensor gradInput, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SoftMarginCriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void MSECriterion_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType target, DoubleTensor.HType output, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <param name="reduction"></param>
		internal static void MSECriterion_updateOutput(DoubleTensor input, DoubleTensor target, DoubleTensor output, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			MSECriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void MSECriterion_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType target, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="reduction"></param>
		internal static void MSECriterion_updateGradInput(DoubleTensor input, DoubleTensor target, DoubleTensor gradOutput, DoubleTensor gradInput, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			MSECriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void MultiLabelMarginCriterion_updateOutput(IntPtr state, DoubleTensor.HType input, LongTensor.HType target, DoubleTensor.HType output, DoubleTensor.HType isTarget, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <param name="isTarget"></param>
		/// <param name="reduction"></param>
		internal static void MultiLabelMarginCriterion_updateOutput(DoubleTensor input, LongTensor target, DoubleTensor output, DoubleTensor isTarget, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (isTarget == null)
				throw new ArgumentNullException (nameof (isTarget));

			MultiLabelMarginCriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, isTarget.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void MultiLabelMarginCriterion_updateGradInput(IntPtr state, DoubleTensor.HType input, LongTensor.HType target, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType isTarget, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="isTarget"></param>
		/// <param name="reduction"></param>
		internal static void MultiLabelMarginCriterion_updateGradInput(DoubleTensor input, LongTensor target, DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor isTarget, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (isTarget == null)
				throw new ArgumentNullException (nameof (isTarget));

			MultiLabelMarginCriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, isTarget.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void MultiMarginCriterion_updateOutput(IntPtr state, DoubleTensor.HType input, LongTensor.HType target, DoubleTensor.HType output, long reduction, int p, DoubleTensor.HType weights, double margin);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <param name="reduction"></param>
		/// <param name="p"></param>
		/// <param name="weights"></param>
		/// <param name="margin"></param>
		internal static void MultiMarginCriterion_updateOutput(DoubleTensor input, LongTensor target, DoubleTensor output, long reduction, int p, DoubleTensor weights, double margin)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weights == null)
				throw new ArgumentNullException (nameof (weights));

			MultiMarginCriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction, p, weights.handle, margin);
		}


		[DllImport ("caffe2")]
		extern static void MultiMarginCriterion_updateGradInput(IntPtr state, DoubleTensor.HType input, LongTensor.HType target, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, long reduction, int p, DoubleTensor.HType weights, double margin);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="reduction"></param>
		/// <param name="p"></param>
		/// <param name="weights"></param>
		/// <param name="margin"></param>
		internal static void MultiMarginCriterion_updateGradInput(DoubleTensor input, LongTensor target, DoubleTensor gradOutput, DoubleTensor gradInput, long reduction, int p, DoubleTensor weights, double margin)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weights == null)
				throw new ArgumentNullException (nameof (weights));

			MultiMarginCriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction, p, weights.handle, margin);
		}


		[DllImport ("caffe2")]
		extern static void RReLU_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, DoubleTensor.HType noise, double lower, double upper, byte train, byte inplace, IntPtr generator);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="noise"></param>
		/// <param name="lower"></param>
		/// <param name="upper"></param>
		/// <param name="train"></param>
		/// <param name="inplace"></param>
		/// <param name="generator"></param>
		internal static void RReLU_updateOutput(DoubleTensor input, DoubleTensor output, DoubleTensor noise, double lower, double upper, bool train, bool inplace, RandomGenerator generator)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (noise == null)
				throw new ArgumentNullException (nameof (noise));

			RReLU_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, noise.handle, lower, upper, (byte)(train ? 1 : 0), (byte)(inplace ? 1 : 0), generator.handle);
		}


		[DllImport ("caffe2")]
		extern static void RReLU_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType noise, double lower, double upper, byte train, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="noise"></param>
		/// <param name="lower"></param>
		/// <param name="upper"></param>
		/// <param name="train"></param>
		/// <param name="inplace"></param>
		internal static void RReLU_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor noise, double lower, double upper, bool train, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (noise == null)
				throw new ArgumentNullException (nameof (noise));

			RReLU_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, noise.handle, lower, upper, (byte)(train ? 1 : 0), (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void Sigmoid_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="output">output tensor</param>
		internal static void Sigmoid_updateOutput(DoubleTensor input, DoubleTensor output)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			Sigmoid_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle);
		}


		[DllImport ("caffe2")]
		extern static void Sigmoid_updateGradInput(IntPtr state, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType output);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput">gradient with regards to module's output</param>
		/// <param name="gradInput">Returned value gradient with regards to input</param>
		/// <param name="output"></param>
		internal static void Sigmoid_updateGradInput(DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor output)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			Sigmoid_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, output.handle);
		}


		[DllImport ("caffe2")]
		extern static void SmoothL1Criterion_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType target, DoubleTensor.HType output, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <param name="reduction"></param>
		internal static void SmoothL1Criterion_updateOutput(DoubleTensor input, DoubleTensor target, DoubleTensor output, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SmoothL1Criterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void SmoothL1Criterion_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType target, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="reduction"></param>
		internal static void SmoothL1Criterion_updateGradInput(DoubleTensor input, DoubleTensor target, DoubleTensor gradOutput, DoubleTensor gradInput, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SmoothL1Criterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void SoftPlus_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, double beta, double threshold);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="beta"></param>
		/// <param name="threshold"></param>
		internal static void SoftPlus_updateOutput(DoubleTensor input, DoubleTensor output, double beta, double threshold)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SoftPlus_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, beta, threshold);
		}


		[DllImport ("caffe2")]
		extern static void SoftPlus_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType output, double beta, double threshold);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="output"></param>
		/// <param name="beta"></param>
		/// <param name="threshold"></param>
		internal static void SoftPlus_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor output, double beta, double threshold)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SoftPlus_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, output.handle, beta, threshold);
		}


		[DllImport ("caffe2")]
		extern static void SoftShrink_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, double lambda);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="lambda"></param>
		internal static void SoftShrink_updateOutput(DoubleTensor input, DoubleTensor output, double lambda)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SoftShrink_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, lambda);
		}


		[DllImport ("caffe2")]
		extern static void SoftShrink_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, double lambda);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="lambda"></param>
		internal static void SoftShrink_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, double lambda)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SoftShrink_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, lambda);
		}


		[DllImport ("caffe2")]
		extern static void IndexLinear_updateOutput(IntPtr state, LongTensor.HType keys, long keysOffset, DoubleTensor.HType values, LongTensor.HType sizes, LongTensor.HType cumSumSizes, DoubleTensor.HType output, DoubleTensor.HType weight, DoubleTensor.HType bias, DoubleTensor.HType normalizedValues, int train);
		/// <summary>
		/// </summary>
		/// <param name="keys"></param>
		/// <param name="keysOffset"></param>
		/// <param name="values"></param>
		/// <param name="sizes"></param>
		/// <param name="cumSumSizes"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="normalizedValues"></param>
		/// <param name="train"></param>
		internal static void IndexLinear_updateOutput(LongTensor keys, long keysOffset, DoubleTensor values, LongTensor sizes, LongTensor cumSumSizes, DoubleTensor output, DoubleTensor weight, DoubleTensor bias, DoubleTensor normalizedValues, int train)
		{
			if (keys == null)
				throw new ArgumentNullException (nameof (keys));
			if (values == null)
				throw new ArgumentNullException (nameof (values));
			if (sizes == null)
				throw new ArgumentNullException (nameof (sizes));
			if (cumSumSizes == null)
				throw new ArgumentNullException (nameof (cumSumSizes));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (normalizedValues == null)
				throw new ArgumentNullException (nameof (normalizedValues));

			IndexLinear_updateOutput(IntPtr.Zero /* state */, keys.handle, keysOffset, values.handle, sizes.handle, cumSumSizes.handle, output.handle, weight.handle, bias.handle, normalizedValues.handle, train);
		}


		[DllImport ("caffe2")]
		extern static void IndexLinear_accGradParameters(IntPtr state, LongTensor.HType keys, long keysOffset, DoubleTensor.HType values, LongTensor.HType sizes, LongTensor.HType cumSumSizes, DoubleTensor.HType gradOutput, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType weight, DoubleTensor.HType bias, DoubleTensor.HType valuesBuffer, double weightDecay, double scale);
		/// <summary>
		/// </summary>
		/// <param name="keys"></param>
		/// <param name="keysOffset"></param>
		/// <param name="values"></param>
		/// <param name="sizes"></param>
		/// <param name="cumSumSizes"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="valuesBuffer"></param>
		/// <param name="weightDecay"></param>
		/// <param name="scale"></param>
		internal static void IndexLinear_accGradParameters(LongTensor keys, long keysOffset, DoubleTensor values, LongTensor sizes, LongTensor cumSumSizes, DoubleTensor gradOutput, DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor weight, DoubleTensor bias, DoubleTensor valuesBuffer, double weightDecay, double scale)
		{
			if (keys == null)
				throw new ArgumentNullException (nameof (keys));
			if (values == null)
				throw new ArgumentNullException (nameof (values));
			if (sizes == null)
				throw new ArgumentNullException (nameof (sizes));
			if (cumSumSizes == null)
				throw new ArgumentNullException (nameof (cumSumSizes));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (valuesBuffer == null)
				throw new ArgumentNullException (nameof (valuesBuffer));

			IndexLinear_accGradParameters(IntPtr.Zero /* state */, keys.handle, keysOffset, values.handle, sizes.handle, cumSumSizes.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, weight.handle, bias.handle, valuesBuffer.handle, weightDecay, scale);
		}


		[DllImport ("caffe2")]
		extern static void IndexLinear_accUpdateGradParameters(IntPtr state, LongTensor.HType keys, long keysOffset, DoubleTensor.HType values, LongTensor.HType sizes, LongTensor.HType cumSumSizes, DoubleTensor.HType gradOutput, DoubleTensor.HType weight, DoubleTensor.HType bias, double weightDecay, double scale);
		/// <summary>
		/// </summary>
		/// <param name="keys"></param>
		/// <param name="keysOffset"></param>
		/// <param name="values"></param>
		/// <param name="sizes"></param>
		/// <param name="cumSumSizes"></param>
		/// <param name="gradOutput"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="weightDecay"></param>
		/// <param name="scale"></param>
		internal static void IndexLinear_accUpdateGradParameters(LongTensor keys, long keysOffset, DoubleTensor values, LongTensor sizes, LongTensor cumSumSizes, DoubleTensor gradOutput, DoubleTensor weight, DoubleTensor bias, double weightDecay, double scale)
		{
			if (keys == null)
				throw new ArgumentNullException (nameof (keys));
			if (values == null)
				throw new ArgumentNullException (nameof (values));
			if (sizes == null)
				throw new ArgumentNullException (nameof (sizes));
			if (cumSumSizes == null)
				throw new ArgumentNullException (nameof (cumSumSizes));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));

			IndexLinear_accUpdateGradParameters(IntPtr.Zero /* state */, keys.handle, keysOffset, values.handle, sizes.handle, cumSumSizes.handle, gradOutput.handle, weight.handle, bias.handle, weightDecay, scale);
		}


		[DllImport ("caffe2")]
		extern static void IndexLinear_updateParameters(IntPtr state, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType weight, DoubleTensor.HType bias, LongTensor.HType runningKeys, LongTensor.HType cumSumSizes, long keysOffset, double weightDecay, double learningRate);
		/// <summary>
		/// </summary>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="runningKeys"></param>
		/// <param name="cumSumSizes"></param>
		/// <param name="keysOffset"></param>
		/// <param name="weightDecay"></param>
		/// <param name="learningRate"></param>
		internal static void IndexLinear_updateParameters(DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor weight, DoubleTensor bias, LongTensor runningKeys, LongTensor cumSumSizes, long keysOffset, double weightDecay, double learningRate)
		{
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (runningKeys == null)
				throw new ArgumentNullException (nameof (runningKeys));
			if (cumSumSizes == null)
				throw new ArgumentNullException (nameof (cumSumSizes));

			IndexLinear_updateParameters(IntPtr.Zero /* state */, gradWeight.handle, gradBias.handle, weight.handle, bias.handle, runningKeys.handle, cumSumSizes.handle, keysOffset, weightDecay, learningRate);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, DoubleTensor.HType weight, DoubleTensor.HType bias);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		internal static void SparseLinear_updateOutput(DoubleTensor input, DoubleTensor output, DoubleTensor weight, DoubleTensor bias)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));

			SparseLinear_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_accGradParameters(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType weight, DoubleTensor.HType bias, double weightDecay, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="weightDecay"></param>
		/// <param name="scale"></param>
		internal static void SparseLinear_accGradParameters(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor weight, DoubleTensor bias, double weightDecay, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));

			SparseLinear_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, weight.handle, bias.handle, weightDecay, scale);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_zeroGradParameters(IntPtr state, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType lastInput);
		/// <summary>
		/// </summary>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="lastInput"></param>
		internal static void SparseLinear_zeroGradParameters(DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor lastInput)
		{
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (lastInput == null)
				throw new ArgumentNullException (nameof (lastInput));

			SparseLinear_zeroGradParameters(IntPtr.Zero /* state */, gradWeight.handle, gradBias.handle, lastInput.handle);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_updateParameters(IntPtr state, DoubleTensor.HType weight, DoubleTensor.HType bias, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType lastInput, double learningRate);
		/// <summary>
		/// </summary>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="lastInput"></param>
		/// <param name="learningRate"></param>
		internal static void SparseLinear_updateParameters(DoubleTensor weight, DoubleTensor bias, DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor lastInput, double learningRate)
		{
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (lastInput == null)
				throw new ArgumentNullException (nameof (lastInput));

			SparseLinear_updateParameters(IntPtr.Zero /* state */, weight.handle, bias.handle, gradWeight.handle, gradBias.handle, lastInput.handle, learningRate);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_legacyUpdateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, DoubleTensor.HType weight, DoubleTensor.HType bias);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		internal static void SparseLinear_legacyUpdateOutput(DoubleTensor input, DoubleTensor output, DoubleTensor weight, DoubleTensor bias)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));

			SparseLinear_legacyUpdateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_legacyAccGradParameters(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType weight, DoubleTensor.HType bias, double weightDecay, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="weightDecay"></param>
		/// <param name="scale"></param>
		internal static void SparseLinear_legacyAccGradParameters(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor weight, DoubleTensor bias, double weightDecay, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));

			SparseLinear_legacyAccGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, weight.handle, bias.handle, weightDecay, scale);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_legacyZeroGradParameters(IntPtr state, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType lastInput);
		/// <summary>
		/// </summary>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="lastInput"></param>
		internal static void SparseLinear_legacyZeroGradParameters(DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor lastInput)
		{
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (lastInput == null)
				throw new ArgumentNullException (nameof (lastInput));

			SparseLinear_legacyZeroGradParameters(IntPtr.Zero /* state */, gradWeight.handle, gradBias.handle, lastInput.handle);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_legacyUpdateParameters(IntPtr state, DoubleTensor.HType weight, DoubleTensor.HType bias, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType lastInput, double learningRate);
		/// <summary>
		/// </summary>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="lastInput"></param>
		/// <param name="learningRate"></param>
		internal static void SparseLinear_legacyUpdateParameters(DoubleTensor weight, DoubleTensor bias, DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor lastInput, double learningRate)
		{
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (lastInput == null)
				throw new ArgumentNullException (nameof (lastInput));

			SparseLinear_legacyUpdateParameters(IntPtr.Zero /* state */, weight.handle, bias.handle, gradWeight.handle, gradBias.handle, lastInput.handle, learningRate);
		}


		[DllImport ("caffe2")]
		extern static void Threshold_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, double threshold, double val, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="threshold"></param>
		/// <param name="val"></param>
		/// <param name="inplace"></param>
		internal static void Threshold_updateOutput(DoubleTensor input, DoubleTensor output, double threshold, double val, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			Threshold_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, threshold, val, (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void Threshold_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, double threshold, double val, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="threshold"></param>
		/// <param name="val"></param>
		/// <param name="inplace"></param>
		internal static void Threshold_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, double threshold, double val, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			Threshold_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, threshold, val, (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void TemporalRowConvolution_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, DoubleTensor.HType weight, DoubleTensor.HType bias, DoubleTensor.HType finput, DoubleTensor.HType fgradInput, int kW, int dW, int padW, byte featFirst);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kW"></param>
		/// <param name="dW"></param>
		/// <param name="padW"></param>
		/// <param name="featFirst"></param>
		internal static void TemporalRowConvolution_updateOutput(DoubleTensor input, DoubleTensor output, DoubleTensor weight, DoubleTensor bias, DoubleTensor finput, DoubleTensor fgradInput, int kW, int dW, int padW, bool featFirst)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			TemporalRowConvolution_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, finput.handle, fgradInput.handle, kW, dW, padW, (byte)(featFirst ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void TemporalRowConvolution_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType weight, DoubleTensor.HType finput, DoubleTensor.HType fgradInput, int kW, int dW, int padW, byte featFirst);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="weight"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kW"></param>
		/// <param name="dW"></param>
		/// <param name="padW"></param>
		/// <param name="featFirst"></param>
		internal static void TemporalRowConvolution_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor weight, DoubleTensor finput, DoubleTensor fgradInput, int kW, int dW, int padW, bool featFirst)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			TemporalRowConvolution_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, weight.handle, finput.handle, fgradInput.handle, kW, dW, padW, (byte)(featFirst ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void TemporalRowConvolution_accGradParameters(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType finput, DoubleTensor.HType fgradInput, int kW, int dW, int padW, byte featFirst, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kW"></param>
		/// <param name="dW"></param>
		/// <param name="padW"></param>
		/// <param name="featFirst"></param>
		/// <param name="scale"></param>
		internal static void TemporalRowConvolution_accGradParameters(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor finput, DoubleTensor fgradInput, int kW, int dW, int padW, bool featFirst, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			TemporalRowConvolution_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, finput.handle, fgradInput.handle, kW, dW, padW, (byte)(featFirst ? 1 : 0), scale);
		}


		[DllImport ("caffe2")]
		extern static void TemporalUpSamplingNearest_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int osizeW);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeW"></param>
		internal static void TemporalUpSamplingNearest_updateOutput(DoubleTensor input, DoubleTensor output, int osizeW)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			TemporalUpSamplingNearest_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeW);
		}


		[DllImport ("caffe2")]
		extern static void TemporalUpSamplingNearest_updateGradInput(IntPtr state, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int isizeB, int isizeC, int isizeW, int osizeW);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="isizeB"></param>
		/// <param name="isizeC"></param>
		/// <param name="isizeW"></param>
		/// <param name="osizeW"></param>
		internal static void TemporalUpSamplingNearest_updateGradInput(DoubleTensor gradOutput, DoubleTensor gradInput, int isizeB, int isizeC, int isizeW, int osizeW)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			TemporalUpSamplingNearest_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, isizeB, isizeC, isizeW, osizeW);
		}


		[DllImport ("caffe2")]
		extern static void TemporalUpSamplingLinear_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int osizeW, byte align_corners);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeW"></param>
		/// <param name="align_corners"></param>
		internal static void TemporalUpSamplingLinear_updateOutput(DoubleTensor input, DoubleTensor output, int osizeW, bool align_corners)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			TemporalUpSamplingLinear_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeW, (byte)(align_corners ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void TemporalUpSamplingLinear_updateGradInput(IntPtr state, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int isizeB, int isizeC, int isizeW, int osizeW, byte align_corners);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="isizeB"></param>
		/// <param name="isizeC"></param>
		/// <param name="isizeW"></param>
		/// <param name="osizeW"></param>
		/// <param name="align_corners"></param>
		internal static void TemporalUpSamplingLinear_updateGradInput(DoubleTensor gradOutput, DoubleTensor gradInput, int isizeB, int isizeC, int isizeW, int osizeW, bool align_corners)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			TemporalUpSamplingLinear_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, isizeB, isizeC, isizeW, osizeW, (byte)(align_corners ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void BatchNormalization_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, DoubleTensor.HType weight, DoubleTensor.HType bias, DoubleTensor.HType running_mean, DoubleTensor.HType running_var, DoubleTensor.HType save_mean, DoubleTensor.HType save_std, byte train, double momentum, double eps);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="running_mean"></param>
		/// <param name="running_var"></param>
		/// <param name="save_mean"></param>
		/// <param name="save_std"></param>
		/// <param name="train"></param>
		/// <param name="momentum"></param>
		/// <param name="eps"></param>
		internal static void BatchNormalization_updateOutput(DoubleTensor input, DoubleTensor output, DoubleTensor weight, DoubleTensor bias, DoubleTensor running_mean, DoubleTensor running_var, DoubleTensor save_mean, DoubleTensor save_std, bool train, double momentum, double eps)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (running_mean == null)
				throw new ArgumentNullException (nameof (running_mean));
			if (running_var == null)
				throw new ArgumentNullException (nameof (running_var));
			if (save_mean == null)
				throw new ArgumentNullException (nameof (save_mean));
			if (save_std == null)
				throw new ArgumentNullException (nameof (save_std));

			BatchNormalization_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, running_mean.handle, running_var.handle, save_mean.handle, save_std.handle, (byte)(train ? 1 : 0), momentum, eps);
		}


		[DllImport ("caffe2")]
		extern static void BatchNormalization_backward(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType weight, DoubleTensor.HType running_mean, DoubleTensor.HType running_var, DoubleTensor.HType save_mean, DoubleTensor.HType save_std, byte train, double scale, double eps);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="weight"></param>
		/// <param name="running_mean"></param>
		/// <param name="running_var"></param>
		/// <param name="save_mean"></param>
		/// <param name="save_std"></param>
		/// <param name="train"></param>
		/// <param name="scale"></param>
		/// <param name="eps"></param>
		internal static void BatchNormalization_backward(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor weight, DoubleTensor running_mean, DoubleTensor running_var, DoubleTensor save_mean, DoubleTensor save_std, bool train, double scale, double eps)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (running_mean == null)
				throw new ArgumentNullException (nameof (running_mean));
			if (running_var == null)
				throw new ArgumentNullException (nameof (running_var));
			if (save_mean == null)
				throw new ArgumentNullException (nameof (save_mean));
			if (save_std == null)
				throw new ArgumentNullException (nameof (save_std));

			BatchNormalization_backward(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, gradWeight.handle, gradBias.handle, weight.handle, running_mean.handle, running_var.handle, save_mean.handle, save_std.handle, (byte)(train ? 1 : 0), scale, eps);
		}


		[DllImport ("caffe2")]
		extern static void SpatialConvolutionMM_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, DoubleTensor.HType weight, DoubleTensor.HType bias, DoubleTensor.HType finput, DoubleTensor.HType fgradInput, int kW, int kH, int dW, int dH, int padW, int padH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		internal static void SpatialConvolutionMM_updateOutput(DoubleTensor input, DoubleTensor output, DoubleTensor weight, DoubleTensor bias, DoubleTensor finput, DoubleTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			SpatialConvolutionMM_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, finput.handle, fgradInput.handle, kW, kH, dW, dH, padW, padH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialConvolutionMM_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType weight, DoubleTensor.HType finput, DoubleTensor.HType fgradInput, int kW, int kH, int dW, int dH, int padW, int padH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="weight"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		internal static void SpatialConvolutionMM_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor weight, DoubleTensor finput, DoubleTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			SpatialConvolutionMM_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, weight.handle, finput.handle, fgradInput.handle, kW, kH, dW, dH, padW, padH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialConvolutionMM_accGradParameters(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType finput, DoubleTensor.HType fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="scale"></param>
		internal static void SpatialConvolutionMM_accGradParameters(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor finput, DoubleTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			SpatialConvolutionMM_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, finput.handle, fgradInput.handle, kW, kH, dW, dH, padW, padH, scale);
		}


		[DllImport ("caffe2")]
		extern static void SpatialAdaptiveMaxPooling_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, LongTensor.HType indices, int osizeW, int osizeH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="indices"></param>
		/// <param name="osizeW"></param>
		/// <param name="osizeH"></param>
		internal static void SpatialAdaptiveMaxPooling_updateOutput(DoubleTensor input, DoubleTensor output, LongTensor indices, int osizeW, int osizeH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			SpatialAdaptiveMaxPooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, indices.handle, osizeW, osizeH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialAdaptiveMaxPooling_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, LongTensor.HType indices);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="indices"></param>
		internal static void SpatialAdaptiveMaxPooling_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, LongTensor indices)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			SpatialAdaptiveMaxPooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, indices.handle);
		}


		[DllImport ("caffe2")]
		extern static void SpatialAdaptiveAveragePooling_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int osizeW, int osizeH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeW"></param>
		/// <param name="osizeH"></param>
		internal static void SpatialAdaptiveAveragePooling_updateOutput(DoubleTensor input, DoubleTensor output, int osizeW, int osizeH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SpatialAdaptiveAveragePooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeW, osizeH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialAdaptiveAveragePooling_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		internal static void SpatialAdaptiveAveragePooling_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SpatialAdaptiveAveragePooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle);
		}


		[DllImport ("caffe2")]
		extern static void SpatialAveragePooling_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int kW, int kH, int dW, int dH, int padW, int padH, byte ceil_mode, byte count_include_pad);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="ceil_mode"></param>
		/// <param name="count_include_pad"></param>
		internal static void SpatialAveragePooling_updateOutput(DoubleTensor input, DoubleTensor output, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SpatialAveragePooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, kW, kH, dW, dH, padW, padH, (byte)(ceil_mode ? 1 : 0), (byte)(count_include_pad ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void SpatialAveragePooling_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int kW, int kH, int dW, int dH, int padW, int padH, byte ceil_mode, byte count_include_pad);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="ceil_mode"></param>
		/// <param name="count_include_pad"></param>
		internal static void SpatialAveragePooling_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SpatialAveragePooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, kW, kH, dW, dH, padW, padH, (byte)(ceil_mode ? 1 : 0), (byte)(count_include_pad ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void SpatialFractionalMaxPooling_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int outputW, int outputH, int kW, int kH, LongTensor.HType indices, DoubleTensor.HType randomSamples);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="outputW"></param>
		/// <param name="outputH"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="indices"></param>
		/// <param name="randomSamples"></param>
		internal static void SpatialFractionalMaxPooling_updateOutput(DoubleTensor input, DoubleTensor output, int outputW, int outputH, int kW, int kH, LongTensor indices, DoubleTensor randomSamples)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));
			if (randomSamples == null)
				throw new ArgumentNullException (nameof (randomSamples));

			SpatialFractionalMaxPooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, outputW, outputH, kW, kH, indices.handle, randomSamples.handle);
		}


		[DllImport ("caffe2")]
		extern static void SpatialFractionalMaxPooling_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int outputW, int outputH, int kW, int kH, LongTensor.HType indices);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="outputW"></param>
		/// <param name="outputH"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="indices"></param>
		internal static void SpatialFractionalMaxPooling_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, int outputW, int outputH, int kW, int kH, LongTensor indices)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			SpatialFractionalMaxPooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, outputW, outputH, kW, kH, indices.handle);
		}


		[DllImport ("caffe2")]
		extern static void SpatialDilatedConvolution_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, DoubleTensor.HType weight, DoubleTensor.HType bias, DoubleTensor.HType columns, DoubleTensor.HType ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="columns"></param>
		/// <param name="ones"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		internal static void SpatialDilatedConvolution_updateOutput(DoubleTensor input, DoubleTensor output, DoubleTensor weight, DoubleTensor bias, DoubleTensor columns, DoubleTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));
			if (ones == null)
				throw new ArgumentNullException (nameof (ones));

			SpatialDilatedConvolution_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, columns.handle, ones.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialDilatedConvolution_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType weight, DoubleTensor.HType columns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="weight"></param>
		/// <param name="columns"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		internal static void SpatialDilatedConvolution_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor weight, DoubleTensor columns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));

			SpatialDilatedConvolution_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, weight.handle, columns.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialDilatedConvolution_accGradParameters(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType columns, DoubleTensor.HType ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="columns"></param>
		/// <param name="ones"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="scale"></param>
		internal static void SpatialDilatedConvolution_accGradParameters(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor columns, DoubleTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));
			if (ones == null)
				throw new ArgumentNullException (nameof (ones));

			SpatialDilatedConvolution_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, columns.handle, ones.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH, scale);
		}


		[DllImport ("caffe2")]
		extern static void SpatialFullDilatedConvolution_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, DoubleTensor.HType weight, DoubleTensor.HType bias, DoubleTensor.HType columns, DoubleTensor.HType ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="columns"></param>
		/// <param name="ones"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="adjW"></param>
		/// <param name="adjH"></param>
		internal static void SpatialFullDilatedConvolution_updateOutput(DoubleTensor input, DoubleTensor output, DoubleTensor weight, DoubleTensor bias, DoubleTensor columns, DoubleTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));
			if (ones == null)
				throw new ArgumentNullException (nameof (ones));

			SpatialFullDilatedConvolution_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, columns.handle, ones.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH, adjW, adjH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialFullDilatedConvolution_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType weight, DoubleTensor.HType columns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="weight"></param>
		/// <param name="columns"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="adjW"></param>
		/// <param name="adjH"></param>
		internal static void SpatialFullDilatedConvolution_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor weight, DoubleTensor columns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));

			SpatialFullDilatedConvolution_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, weight.handle, columns.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH, adjW, adjH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialFullDilatedConvolution_accGradParameters(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType columns, DoubleTensor.HType ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="columns"></param>
		/// <param name="ones"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="adjW"></param>
		/// <param name="adjH"></param>
		/// <param name="scale"></param>
		internal static void SpatialFullDilatedConvolution_accGradParameters(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor columns, DoubleTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));
			if (ones == null)
				throw new ArgumentNullException (nameof (ones));

			SpatialFullDilatedConvolution_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, columns.handle, ones.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH, adjW, adjH, scale);
		}


		[DllImport ("caffe2")]
		extern static void SpatialDilatedMaxPooling_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, LongTensor.HType indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, byte ceil_mode);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="indices"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="ceil_mode"></param>
		internal static void SpatialDilatedMaxPooling_updateOutput(DoubleTensor input, DoubleTensor output, LongTensor indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			SpatialDilatedMaxPooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, indices.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH, (byte)(ceil_mode ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void SpatialDilatedMaxPooling_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, LongTensor.HType indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, byte ceil_mode);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="indices"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="ceil_mode"></param>
		internal static void SpatialDilatedMaxPooling_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, LongTensor indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			SpatialDilatedMaxPooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, indices.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH, (byte)(ceil_mode ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void SpatialMaxUnpooling_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, LongTensor.HType indices, int owidth, int oheight);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="indices"></param>
		/// <param name="owidth"></param>
		/// <param name="oheight"></param>
		internal static void SpatialMaxUnpooling_updateOutput(DoubleTensor input, DoubleTensor output, LongTensor indices, int owidth, int oheight)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			SpatialMaxUnpooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, indices.handle, owidth, oheight);
		}


		[DllImport ("caffe2")]
		extern static void SpatialMaxUnpooling_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, LongTensor.HType indices, int owidth, int oheight);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="indices"></param>
		/// <param name="owidth"></param>
		/// <param name="oheight"></param>
		internal static void SpatialMaxUnpooling_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, LongTensor indices, int owidth, int oheight)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			SpatialMaxUnpooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, indices.handle, owidth, oheight);
		}


		[DllImport ("caffe2")]
		extern static void SpatialUpSamplingNearest_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int osizeH, int osizeW);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		internal static void SpatialUpSamplingNearest_updateOutput(DoubleTensor input, DoubleTensor output, int osizeH, int osizeW)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SpatialUpSamplingNearest_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeH, osizeW);
		}


		[DllImport ("caffe2")]
		extern static void SpatialUpSamplingNearest_updateGradInput(IntPtr state, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int isizeB, int isizeC, int isizeH, int isizeW, int osizeH, int osizeW);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="isizeB"></param>
		/// <param name="isizeC"></param>
		/// <param name="isizeH"></param>
		/// <param name="isizeW"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		internal static void SpatialUpSamplingNearest_updateGradInput(DoubleTensor gradOutput, DoubleTensor gradInput, int isizeB, int isizeC, int isizeH, int isizeW, int osizeH, int osizeW)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SpatialUpSamplingNearest_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, isizeB, isizeC, isizeH, isizeW, osizeH, osizeW);
		}


		[DllImport ("caffe2")]
		extern static void SpatialUpSamplingBilinear_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int osizeH, int osizeW, byte align_corners);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		/// <param name="align_corners"></param>
		internal static void SpatialUpSamplingBilinear_updateOutput(DoubleTensor input, DoubleTensor output, int osizeH, int osizeW, bool align_corners)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SpatialUpSamplingBilinear_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeH, osizeW, (byte)(align_corners ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void SpatialUpSamplingBilinear_updateGradInput(IntPtr state, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int isizeB, int isizeC, int isizeH, int isizeW, int osizeH, int osizeW, byte align_corners);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="isizeB"></param>
		/// <param name="isizeC"></param>
		/// <param name="isizeH"></param>
		/// <param name="isizeW"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		/// <param name="align_corners"></param>
		internal static void SpatialUpSamplingBilinear_updateGradInput(DoubleTensor gradOutput, DoubleTensor gradInput, int isizeB, int isizeC, int isizeH, int isizeW, int osizeH, int osizeW, bool align_corners)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SpatialUpSamplingBilinear_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, isizeB, isizeC, isizeH, isizeW, osizeH, osizeW, (byte)(align_corners ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void unfolded_acc(DoubleTensor.HType finput, DoubleTensor.HType input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int osizeW, int outputHeight);
		/// <summary>
		/// </summary>
		/// <param name="finput"></param>
		/// <param name="input"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="nInputPlane"></param>
		/// <param name="inputWidth"></param>
		/// <param name="inputHeight"></param>
		/// <param name="osizeW"></param>
		/// <param name="outputHeight"></param>
		internal static void unfolded_acc(DoubleTensor finput, DoubleTensor input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int osizeW, int outputHeight)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));

			unfolded_acc(finput.handle, input.handle, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, osizeW, outputHeight);
		}


		[DllImport ("caffe2")]
		extern static void unfolded_copy(DoubleTensor.HType finput, DoubleTensor.HType input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int outputWidth, int outputHeight);
		/// <summary>
		/// </summary>
		/// <param name="finput"></param>
		/// <param name="input"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="nInputPlane"></param>
		/// <param name="inputWidth"></param>
		/// <param name="inputHeight"></param>
		/// <param name="outputWidth"></param>
		/// <param name="outputHeight"></param>
		internal static void unfolded_copy(DoubleTensor finput, DoubleTensor input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));

			unfolded_copy(finput.handle, input.handle, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricAveragePooling_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, byte ceil_mode, byte count_include_pad);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padT"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="ceil_mode"></param>
		/// <param name="count_include_pad"></param>
		internal static void VolumetricAveragePooling_updateOutput(DoubleTensor input, DoubleTensor output, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, bool ceil_mode, bool count_include_pad)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			VolumetricAveragePooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, kT, kW, kH, dT, dW, dH, padT, padW, padH, (byte)(ceil_mode ? 1 : 0), (byte)(count_include_pad ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void VolumetricAveragePooling_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, byte ceil_mode, byte count_include_pad);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padT"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="ceil_mode"></param>
		/// <param name="count_include_pad"></param>
		internal static void VolumetricAveragePooling_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, bool ceil_mode, bool count_include_pad)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			VolumetricAveragePooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, kT, kW, kH, dT, dW, dH, padT, padW, padH, (byte)(ceil_mode ? 1 : 0), (byte)(count_include_pad ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void VolumetricDilatedConvolution_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, DoubleTensor.HType weight, DoubleTensor.HType bias, DoubleTensor.HType columns, DoubleTensor.HType ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="columns"></param>
		/// <param name="ones"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padT"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		internal static void VolumetricDilatedConvolution_updateOutput(DoubleTensor input, DoubleTensor output, DoubleTensor weight, DoubleTensor bias, DoubleTensor columns, DoubleTensor ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));
			if (ones == null)
				throw new ArgumentNullException (nameof (ones));

			VolumetricDilatedConvolution_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, columns.handle, ones.handle, kT, kW, kH, dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricDilatedConvolution_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType weight, DoubleTensor.HType columns, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="weight"></param>
		/// <param name="columns"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padT"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		internal static void VolumetricDilatedConvolution_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor weight, DoubleTensor columns, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));

			VolumetricDilatedConvolution_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, weight.handle, columns.handle, kT, kW, kH, dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricDilatedConvolution_accGradParameters(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType columns, DoubleTensor.HType ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="columns"></param>
		/// <param name="ones"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padT"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="scale"></param>
		internal static void VolumetricDilatedConvolution_accGradParameters(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor columns, DoubleTensor ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));
			if (ones == null)
				throw new ArgumentNullException (nameof (ones));

			VolumetricDilatedConvolution_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, columns.handle, ones.handle, kT, kW, kH, dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH, scale);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricFullDilatedConvolution_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, DoubleTensor.HType weight, DoubleTensor.HType bias, DoubleTensor.HType finput, DoubleTensor.HType fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="aT"></param>
		/// <param name="aW"></param>
		/// <param name="aH"></param>
		internal static void VolumetricFullDilatedConvolution_updateOutput(DoubleTensor input, DoubleTensor output, DoubleTensor weight, DoubleTensor bias, DoubleTensor finput, DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			VolumetricFullDilatedConvolution_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, finput.handle, fgradInput.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricFullDilatedConvolution_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType weight, DoubleTensor.HType finput, DoubleTensor.HType fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="weight"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="aT"></param>
		/// <param name="aW"></param>
		/// <param name="aH"></param>
		internal static void VolumetricFullDilatedConvolution_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor weight, DoubleTensor finput, DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			VolumetricFullDilatedConvolution_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, weight.handle, finput.handle, fgradInput.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricFullDilatedConvolution_accGradParameters(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType finput, DoubleTensor.HType fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="aT"></param>
		/// <param name="aW"></param>
		/// <param name="aH"></param>
		/// <param name="scale"></param>
		internal static void VolumetricFullDilatedConvolution_accGradParameters(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor finput, DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			VolumetricFullDilatedConvolution_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, finput.handle, fgradInput.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH, scale);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricDilatedMaxPooling_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, LongTensor.HType indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, byte ceilMode);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="indices"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="ceilMode"></param>
		internal static void VolumetricDilatedMaxPooling_updateOutput(DoubleTensor input, DoubleTensor output, LongTensor indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			VolumetricDilatedMaxPooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, indices.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, (byte)(ceilMode ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void VolumetricDilatedMaxPooling_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, LongTensor.HType indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, byte ceilMode);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="indices"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="ceilMode"></param>
		internal static void VolumetricDilatedMaxPooling_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, LongTensor indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			VolumetricDilatedMaxPooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, indices.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, (byte)(ceilMode ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void VolumetricMaxUnpooling_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, LongTensor.HType indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="indices"></param>
		/// <param name="oT"></param>
		/// <param name="oW"></param>
		/// <param name="oH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		internal static void VolumetricMaxUnpooling_updateOutput(DoubleTensor input, DoubleTensor output, LongTensor indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			VolumetricMaxUnpooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, indices.handle, oT, oW, oH, dT, dW, dH, pT, pW, pH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricMaxUnpooling_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, LongTensor.HType indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="indices"></param>
		/// <param name="oT"></param>
		/// <param name="oW"></param>
		/// <param name="oH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		internal static void VolumetricMaxUnpooling_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, LongTensor indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			VolumetricMaxUnpooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, indices.handle, oT, oW, oH, dT, dW, dH, pT, pW, pH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricAdaptiveAveragePooling_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int osizeT, int osizeW, int osizeH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeT"></param>
		/// <param name="osizeW"></param>
		/// <param name="osizeH"></param>
		internal static void VolumetricAdaptiveAveragePooling_updateOutput(DoubleTensor input, DoubleTensor output, int osizeT, int osizeW, int osizeH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			VolumetricAdaptiveAveragePooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeT, osizeW, osizeH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricAdaptiveAveragePooling_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		internal static void VolumetricAdaptiveAveragePooling_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			VolumetricAdaptiveAveragePooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricAdaptiveMaxPooling_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, LongTensor.HType indices, int osizeT, int osizeW, int osizeH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="indices"></param>
		/// <param name="osizeT"></param>
		/// <param name="osizeW"></param>
		/// <param name="osizeH"></param>
		internal static void VolumetricAdaptiveMaxPooling_updateOutput(DoubleTensor input, DoubleTensor output, LongTensor indices, int osizeT, int osizeW, int osizeH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			VolumetricAdaptiveMaxPooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, indices.handle, osizeT, osizeW, osizeH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricAdaptiveMaxPooling_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, LongTensor.HType indices);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="indices"></param>
		internal static void VolumetricAdaptiveMaxPooling_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, LongTensor indices)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			VolumetricAdaptiveMaxPooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, indices.handle);
		}


		[DllImport ("caffe2")]
		extern static void SpatialReflectionPadding_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int pad_left, int pad_right, int pad_top, int pad_bottom);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		/// <param name="pad_top"></param>
		/// <param name="pad_bottom"></param>
		internal static void SpatialReflectionPadding_updateOutput(DoubleTensor input, DoubleTensor output, int pad_left, int pad_right, int pad_top, int pad_bottom)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SpatialReflectionPadding_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, pad_left, pad_right, pad_top, pad_bottom);
		}


		[DllImport ("caffe2")]
		extern static void SpatialReflectionPadding_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		/// <param name="pad_top"></param>
		/// <param name="pad_bottom"></param>
		internal static void SpatialReflectionPadding_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SpatialReflectionPadding_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, pad_left, pad_right, pad_top, pad_bottom);
		}


		[DllImport ("caffe2")]
		extern static void SpatialReplicationPadding_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int pad_left, int pad_right, int pad_top, int pad_bottom);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		/// <param name="pad_top"></param>
		/// <param name="pad_bottom"></param>
		internal static void SpatialReplicationPadding_updateOutput(DoubleTensor input, DoubleTensor output, int pad_left, int pad_right, int pad_top, int pad_bottom)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SpatialReplicationPadding_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, pad_left, pad_right, pad_top, pad_bottom);
		}


		[DllImport ("caffe2")]
		extern static void SpatialReplicationPadding_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		/// <param name="pad_top"></param>
		/// <param name="pad_bottom"></param>
		internal static void SpatialReplicationPadding_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SpatialReplicationPadding_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, pad_left, pad_right, pad_top, pad_bottom);
		}


		[DllImport ("caffe2")]
		extern static void FeatureLPPooling_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, double power, int width, int stride, byte batchMode);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="power"></param>
		/// <param name="width"></param>
		/// <param name="stride"></param>
		/// <param name="batchMode"></param>
		internal static void FeatureLPPooling_updateOutput(DoubleTensor input, DoubleTensor output, double power, int width, int stride, bool batchMode)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			FeatureLPPooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, power, width, stride, (byte)(batchMode ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void FeatureLPPooling_updateGradInput(IntPtr state, DoubleTensor.HType gradOutput, DoubleTensor.HType input, DoubleTensor.HType output, DoubleTensor.HType gradInput, double power, int width, int stride, byte batchMode);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="gradInput"></param>
		/// <param name="power"></param>
		/// <param name="width"></param>
		/// <param name="stride"></param>
		/// <param name="batchMode"></param>
		internal static void FeatureLPPooling_updateGradInput(DoubleTensor gradOutput, DoubleTensor input, DoubleTensor output, DoubleTensor gradInput, double power, int width, int stride, bool batchMode)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			FeatureLPPooling_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, input.handle, output.handle, gradInput.handle, power, width, stride, (byte)(batchMode ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void VolumetricReplicationPadding_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_back);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		/// <param name="pad_top"></param>
		/// <param name="pad_bottom"></param>
		/// <param name="pad_front"></param>
		/// <param name="pad_back"></param>
		internal static void VolumetricReplicationPadding_updateOutput(DoubleTensor input, DoubleTensor output, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_back)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			VolumetricReplicationPadding_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricReplicationPadding_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_back);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		/// <param name="pad_top"></param>
		/// <param name="pad_bottom"></param>
		/// <param name="pad_front"></param>
		/// <param name="pad_back"></param>
		internal static void VolumetricReplicationPadding_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_back)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			VolumetricReplicationPadding_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricUpSamplingNearest_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int osizeT, int osizeH, int osizeW);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeT"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		internal static void VolumetricUpSamplingNearest_updateOutput(DoubleTensor input, DoubleTensor output, int osizeT, int osizeH, int osizeW)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			VolumetricUpSamplingNearest_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeT, osizeH, osizeW);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricUpSamplingNearest_updateGradInput(IntPtr state, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int isizeB, int isizeC, int isizeT, int isizeH, int isizeW, int osizeT, int osizeH, int osizeW);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="isizeB"></param>
		/// <param name="isizeC"></param>
		/// <param name="isizeT"></param>
		/// <param name="isizeH"></param>
		/// <param name="isizeW"></param>
		/// <param name="osizeT"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		internal static void VolumetricUpSamplingNearest_updateGradInput(DoubleTensor gradOutput, DoubleTensor gradInput, int isizeB, int isizeC, int isizeT, int isizeH, int isizeW, int osizeT, int osizeH, int osizeW)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			VolumetricUpSamplingNearest_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, isizeB, isizeC, isizeT, isizeH, isizeW, osizeT, osizeH, osizeW);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricUpSamplingTrilinear_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int osizeT, int osizeH, int osizeW, byte align_corners);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeT"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		/// <param name="align_corners"></param>
		internal static void VolumetricUpSamplingTrilinear_updateOutput(DoubleTensor input, DoubleTensor output, int osizeT, int osizeH, int osizeW, bool align_corners)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			VolumetricUpSamplingTrilinear_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeT, osizeH, osizeW, (byte)(align_corners ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void VolumetricUpSamplingTrilinear_updateGradInput(IntPtr state, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int isizeB, int isizeC, int isizeT, int isizeH, int isizeW, int osizeT, int osizeH, int osizeW, byte align_corners);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="isizeB"></param>
		/// <param name="isizeC"></param>
		/// <param name="isizeT"></param>
		/// <param name="isizeH"></param>
		/// <param name="isizeW"></param>
		/// <param name="osizeT"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		/// <param name="align_corners"></param>
		internal static void VolumetricUpSamplingTrilinear_updateGradInput(DoubleTensor gradOutput, DoubleTensor gradInput, int isizeB, int isizeC, int isizeT, int isizeH, int isizeW, int osizeT, int osizeH, int osizeW, bool align_corners)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			VolumetricUpSamplingTrilinear_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, isizeB, isizeC, isizeT, isizeH, isizeW, osizeT, osizeH, osizeW, (byte)(align_corners ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void TemporalReflectionPadding_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int pad_left, int pad_right);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		internal static void TemporalReflectionPadding_updateOutput(DoubleTensor input, DoubleTensor output, int pad_left, int pad_right)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			TemporalReflectionPadding_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, pad_left, pad_right);
		}


		[DllImport ("caffe2")]
		extern static void TemporalReflectionPadding_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int pad_left, int pad_right);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		internal static void TemporalReflectionPadding_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, int pad_left, int pad_right)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			TemporalReflectionPadding_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, pad_left, pad_right);
		}


		[DllImport ("caffe2")]
		extern static void TemporalReplicationPadding_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, int pad_left, int pad_right);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		internal static void TemporalReplicationPadding_updateOutput(DoubleTensor input, DoubleTensor output, int pad_left, int pad_right)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			TemporalReplicationPadding_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, pad_left, pad_right);
		}


		[DllImport ("caffe2")]
		extern static void TemporalReplicationPadding_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, int pad_left, int pad_right);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		internal static void TemporalReplicationPadding_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, int pad_left, int pad_right)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			TemporalReplicationPadding_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, pad_left, pad_right);
		}


		[DllImport ("caffe2")]
		extern static void Tanh_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="output">Returned value output tensor</param>
		internal static void Tanh_updateOutput(DoubleTensor input, DoubleTensor output)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			Tanh_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle);
		}


		[DllImport ("caffe2")]
		extern static void Tanh_updateGradInput(IntPtr state, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType output);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput">gradient with regards to module's output</param>
		/// <param name="gradInput">Returned value gradient with regards to the input</param>
		/// <param name="output"></param>
		internal static void Tanh_updateGradInput(DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor output)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			Tanh_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, output.handle);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricConvolutionMM_updateOutput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType output, DoubleTensor.HType weight, DoubleTensor.HType bias, DoubleTensor.HType finput, DoubleTensor.HType fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		internal static void VolumetricConvolutionMM_updateOutput(DoubleTensor input, DoubleTensor output, DoubleTensor weight, DoubleTensor bias, DoubleTensor finput, DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			VolumetricConvolutionMM_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, finput.handle, fgradInput.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricConvolutionMM_updateGradInput(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, DoubleTensor.HType weight, DoubleTensor.HType finput, DoubleTensor.HType fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="weight"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		internal static void VolumetricConvolutionMM_updateGradInput(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradInput, DoubleTensor weight, DoubleTensor finput, DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			VolumetricConvolutionMM_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, weight.handle, finput.handle, fgradInput.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricConvolutionMM_accGradParameters(IntPtr state, DoubleTensor.HType input, DoubleTensor.HType gradOutput, DoubleTensor.HType gradWeight, DoubleTensor.HType gradBias, DoubleTensor.HType finput, DoubleTensor.HType fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		/// <param name="scale"></param>
		internal static void VolumetricConvolutionMM_accGradParameters(DoubleTensor input, DoubleTensor gradOutput, DoubleTensor gradWeight, DoubleTensor gradBias, DoubleTensor finput, DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			VolumetricConvolutionMM_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, finput.handle, fgradInput.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH, scale);
		}


		[DllImport ("caffe2")]
		extern static void SpatialClassNLLCriterion_updateOutput(IntPtr state, DoubleTensor.HType input, LongTensor.HType target, DoubleTensor.HType output, long reduction, DoubleTensor.HType weights, DoubleTensor.HType total_weight, long ignore_index);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <param name="reduction"></param>
		/// <param name="weights"></param>
		/// <param name="total_weight"></param>
		/// <param name="ignore_index"></param>
		internal static void SpatialClassNLLCriterion_updateOutput(DoubleTensor input, LongTensor target, DoubleTensor output, long reduction, DoubleTensor weights, DoubleTensor total_weight, long ignore_index)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weights == null)
				throw new ArgumentNullException (nameof (weights));
			if (total_weight == null)
				throw new ArgumentNullException (nameof (total_weight));

			SpatialClassNLLCriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction, weights.handle, total_weight.handle, ignore_index);
		}


		[DllImport ("caffe2")]
		extern static void SpatialClassNLLCriterion_updateGradInput(IntPtr state, DoubleTensor.HType input, LongTensor.HType target, DoubleTensor.HType gradOutput, DoubleTensor.HType gradInput, long reduction, DoubleTensor.HType weights, DoubleTensor.HType total_weight, long ignore_index);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="reduction"></param>
		/// <param name="weights"></param>
		/// <param name="total_weight"></param>
		/// <param name="ignore_index"></param>
		internal static void SpatialClassNLLCriterion_updateGradInput(DoubleTensor input, LongTensor target, DoubleTensor gradOutput, DoubleTensor gradInput, long reduction, DoubleTensor weights, DoubleTensor total_weight, long ignore_index)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weights == null)
				throw new ArgumentNullException (nameof (weights));
			if (total_weight == null)
				throw new ArgumentNullException (nameof (total_weight));

			SpatialClassNLLCriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction, weights.handle, total_weight.handle, ignore_index);
		}


	}
	public partial class FloatTensor {
		[DllImport ("caffe2")]
		extern static void AbsCriterion_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType target, FloatTensor.HType output, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="target">tensor with target values</param>
		/// <param name="output">Returned value a one-element tensor with loss</param>
		/// <param name="reduction"></param>
		internal static void AbsCriterion_updateOutput(FloatTensor input, FloatTensor target, FloatTensor output, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			AbsCriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void AbsCriterion_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType target, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="target">tensor with target values</param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput">Returned value gradient with regards to input</param>
		/// <param name="reduction"></param>
		internal static void AbsCriterion_updateGradInput(FloatTensor input, FloatTensor target, FloatTensor gradOutput, FloatTensor gradInput, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			AbsCriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void BCECriterion_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType target, FloatTensor.HType output, long reduction, FloatTensor.HType weights);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <param name="reduction"></param>
		/// <param name="weights"></param>
		internal static void BCECriterion_updateOutput(FloatTensor input, FloatTensor target, FloatTensor output, long reduction, FloatTensor weights)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weights == null)
				throw new ArgumentNullException (nameof (weights));

			BCECriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction, weights.handle);
		}


		[DllImport ("caffe2")]
		extern static void BCECriterion_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType target, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, long reduction, FloatTensor.HType weights);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="reduction"></param>
		/// <param name="weights"></param>
		internal static void BCECriterion_updateGradInput(FloatTensor input, FloatTensor target, FloatTensor gradOutput, FloatTensor gradInput, long reduction, FloatTensor weights)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weights == null)
				throw new ArgumentNullException (nameof (weights));

			BCECriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction, weights.handle);
		}


		[DllImport ("caffe2")]
		extern static void ClassNLLCriterion_updateOutput(IntPtr state, FloatTensor.HType input, LongTensor.HType target, FloatTensor.HType output, long reduction, FloatTensor.HType weights, FloatTensor.HType total_weight, long ignore_index);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor (1D/2D)</param>
		/// <param name="target">tensor containing indexes of target classes</param>
		/// <param name="output">Returned value a one-element tensor with loss</param>
		/// <param name="reduction"></param>
		/// <param name="weights">Optional, can be null. class weights</param>
		/// <param name="total_weight">A buffer.  The _updateGradInput and _accGradParameters methods should get the same buffers that were used in _updateOutput call.</param>
		/// <param name="ignore_index"></param>
		internal static void ClassNLLCriterion_updateOutput(FloatTensor input, LongTensor target, FloatTensor output, long reduction, FloatTensor weights, FloatTensor total_weight, long ignore_index)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (total_weight == null)
				throw new ArgumentNullException (nameof (total_weight));

			ClassNLLCriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction, weights == null ? null : weights.handle, total_weight.handle, ignore_index);
		}


		[DllImport ("caffe2")]
		extern static void ClassNLLCriterion_updateGradInput(IntPtr state, FloatTensor.HType input, LongTensor.HType target, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, long reduction, FloatTensor.HType weights, FloatTensor.HType total_weight, long ignore_index);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor (1D/2D)</param>
		/// <param name="target">tensor containing indexes of target classes</param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput">Returned value gradient with regards to input</param>
		/// <param name="reduction"></param>
		/// <param name="weights">Optional, can be null. class weights</param>
		/// <param name="total_weight">A buffer.  The _updateGradInput and _accGradParameters methods should get the same buffers that were used in _updateOutput call.</param>
		/// <param name="ignore_index"></param>
		internal static void ClassNLLCriterion_updateGradInput(FloatTensor input, LongTensor target, FloatTensor gradOutput, FloatTensor gradInput, long reduction, FloatTensor weights, FloatTensor total_weight, long ignore_index)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (total_weight == null)
				throw new ArgumentNullException (nameof (total_weight));

			ClassNLLCriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction, weights == null ? null : weights.handle, total_weight.handle, ignore_index);
		}


		[DllImport ("caffe2")]
		extern static void ELU_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, double alpha, double scale, double input_scale, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="output">Returned value ELU output</param>
		/// <param name="alpha">an ELU parameter (as in paper)</param>
		/// <param name="scale"></param>
		/// <param name="input_scale"></param>
		/// <param name="inplace"></param>
		internal static void ELU_updateOutput(FloatTensor input, FloatTensor output, double alpha, double scale, double input_scale, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			ELU_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, alpha, scale, input_scale, (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void ELU_updateGradInput(IntPtr state, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType output, double alpha, double scale, double input_scale);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput">gradient with regards to output</param>
		/// <param name="gradInput">Returned value gradient with regards to input</param>
		/// <param name="output">output from a forward pass</param>
		/// <param name="alpha">an ELU parameter (as in paper)</param>
		/// <param name="scale"></param>
		/// <param name="input_scale"></param>
		internal static void ELU_updateGradInput(FloatTensor gradOutput, FloatTensor gradInput, FloatTensor output, double alpha, double scale, double input_scale)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			ELU_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, output.handle, alpha, scale, input_scale);
		}


		[DllImport ("caffe2")]
		extern static void GatedLinear_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int dim);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="dim"></param>
		internal static void GatedLinear_updateOutput(FloatTensor input, FloatTensor output, int dim)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			GatedLinear_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, dim);
		}


		[DllImport ("caffe2")]
		extern static void GatedLinear_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int dim);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="dim"></param>
		internal static void GatedLinear_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, int dim)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			GatedLinear_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, dim);
		}


		[DllImport ("caffe2")]
		extern static void HardTanh_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, double min_val, double max_val, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="output">Returned value output tensor</param>
		/// <param name="min_val">lower threshold</param>
		/// <param name="max_val">upper threshold</param>
		/// <param name="inplace"></param>
		internal static void HardTanh_updateOutput(FloatTensor input, FloatTensor output, double min_val, double max_val, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			HardTanh_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, min_val, max_val, (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void HardTanh_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, double min_val, double max_val, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="gradOutput">gradient with regards to module's output</param>
		/// <param name="gradInput">Returned value gradient with regards to the input</param>
		/// <param name="min_val">lower threshold</param>
		/// <param name="max_val">upper threshold</param>
		/// <param name="inplace"></param>
		internal static void HardTanh_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, double min_val, double max_val, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			HardTanh_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, min_val, max_val, (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void Im2Col_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="kH"></param>
		/// <param name="kW"></param>
		/// <param name="dH"></param>
		/// <param name="dW"></param>
		/// <param name="padH"></param>
		/// <param name="padW"></param>
		/// <param name="sH"></param>
		/// <param name="sW"></param>
		internal static void Im2Col_updateOutput(FloatTensor input, FloatTensor output, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			Im2Col_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, kH, kW, dH, dW, padH, padW, sH, sW);
		}


		[DllImport ("caffe2")]
		extern static void Im2Col_updateGradInput(IntPtr state, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, long inputHeight, long inputWidth, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="inputHeight"></param>
		/// <param name="inputWidth"></param>
		/// <param name="kH"></param>
		/// <param name="kW"></param>
		/// <param name="dH"></param>
		/// <param name="dW"></param>
		/// <param name="padH"></param>
		/// <param name="padW"></param>
		/// <param name="sH"></param>
		/// <param name="sW"></param>
		internal static void Im2Col_updateGradInput(FloatTensor gradOutput, FloatTensor gradInput, long inputHeight, long inputWidth, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			Im2Col_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, inputHeight, inputWidth, kH, kW, dH, dW, padH, padW, sH, sW);
		}


		[DllImport ("caffe2")]
		extern static void Col2Im_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, long outputHeight, long outputWidth, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="outputHeight"></param>
		/// <param name="outputWidth"></param>
		/// <param name="kH"></param>
		/// <param name="kW"></param>
		/// <param name="dH"></param>
		/// <param name="dW"></param>
		/// <param name="padH"></param>
		/// <param name="padW"></param>
		/// <param name="sH"></param>
		/// <param name="sW"></param>
		internal static void Col2Im_updateOutput(FloatTensor input, FloatTensor output, long outputHeight, long outputWidth, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			Col2Im_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, outputHeight, outputWidth, kH, kW, dH, dW, padH, padW, sH, sW);
		}


		[DllImport ("caffe2")]
		extern static void Col2Im_updateGradInput(IntPtr state, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="kH"></param>
		/// <param name="kW"></param>
		/// <param name="dH"></param>
		/// <param name="dW"></param>
		/// <param name="padH"></param>
		/// <param name="padW"></param>
		/// <param name="sH"></param>
		/// <param name="sW"></param>
		internal static void Col2Im_updateGradInput(FloatTensor gradOutput, FloatTensor gradInput, long kH, long kW, long dH, long dW, long padH, long padW, long sH, long sW)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			Col2Im_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, kH, kW, dH, dW, padH, padW, sH, sW);
		}


		[DllImport ("caffe2")]
		extern static void LeakyReLU_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, double negval, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input">**[MODIFIED]** input tensor</param>
		/// <param name="output">Returned value output tensor</param>
		/// <param name="negval">negative part slope</param>
		/// <param name="inplace">if true, modifies the input tensor and sets the output tensor on it (no additional memory is allocated)</param>
		internal static void LeakyReLU_updateOutput(FloatTensor input, FloatTensor output, double negval, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			LeakyReLU_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, negval, (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void LeakyReLU_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, double negval, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="gradOutput">**[MODIFIED]** gradient with regards to module's output</param>
		/// <param name="gradInput">Returned value gradient with regards to the input</param>
		/// <param name="negval">negative part slope</param>
		/// <param name="inplace">if true, modifies gradOutput and sets gradInput onto it (no additional memory is allocated)</param>
		internal static void LeakyReLU_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, double negval, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			LeakyReLU_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, negval, (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void LogSigmoid_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, FloatTensor.HType buffer);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="output">output tensor</param>
		/// <param name="buffer">A buffer.  The _updateGradInput and _accGradParameters methods should get the same buffers that were used in _updateOutput call.</param>
		internal static void LogSigmoid_updateOutput(FloatTensor input, FloatTensor output, FloatTensor buffer)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (buffer == null)
				throw new ArgumentNullException (nameof (buffer));

			LogSigmoid_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, buffer.handle);
		}


		[DllImport ("caffe2")]
		extern static void LogSigmoid_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType buffer);
		/// <summary>
		/// </summary>
		/// <param name="input">input</param>
		/// <param name="gradOutput">gradient with regards to module's output</param>
		/// <param name="gradInput">Returned value gradient with regards to input</param>
		/// <param name="buffer">A buffer.  The _updateGradInput and _accGradParameters methods should get the same buffers that were used in _updateOutput call.</param>
		internal static void LogSigmoid_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, FloatTensor buffer)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (buffer == null)
				throw new ArgumentNullException (nameof (buffer));

			LogSigmoid_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, buffer.handle);
		}


		[DllImport ("caffe2")]
		extern static void SoftMarginCriterion_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType target, FloatTensor.HType output, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <param name="reduction"></param>
		internal static void SoftMarginCriterion_updateOutput(FloatTensor input, FloatTensor target, FloatTensor output, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SoftMarginCriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void SoftMarginCriterion_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType target, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="reduction"></param>
		internal static void SoftMarginCriterion_updateGradInput(FloatTensor input, FloatTensor target, FloatTensor gradOutput, FloatTensor gradInput, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SoftMarginCriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void MSECriterion_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType target, FloatTensor.HType output, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <param name="reduction"></param>
		internal static void MSECriterion_updateOutput(FloatTensor input, FloatTensor target, FloatTensor output, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			MSECriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void MSECriterion_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType target, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="reduction"></param>
		internal static void MSECriterion_updateGradInput(FloatTensor input, FloatTensor target, FloatTensor gradOutput, FloatTensor gradInput, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			MSECriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void MultiLabelMarginCriterion_updateOutput(IntPtr state, FloatTensor.HType input, LongTensor.HType target, FloatTensor.HType output, FloatTensor.HType isTarget, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <param name="isTarget"></param>
		/// <param name="reduction"></param>
		internal static void MultiLabelMarginCriterion_updateOutput(FloatTensor input, LongTensor target, FloatTensor output, FloatTensor isTarget, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (isTarget == null)
				throw new ArgumentNullException (nameof (isTarget));

			MultiLabelMarginCriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, isTarget.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void MultiLabelMarginCriterion_updateGradInput(IntPtr state, FloatTensor.HType input, LongTensor.HType target, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType isTarget, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="isTarget"></param>
		/// <param name="reduction"></param>
		internal static void MultiLabelMarginCriterion_updateGradInput(FloatTensor input, LongTensor target, FloatTensor gradOutput, FloatTensor gradInput, FloatTensor isTarget, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (isTarget == null)
				throw new ArgumentNullException (nameof (isTarget));

			MultiLabelMarginCriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, isTarget.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void MultiMarginCriterion_updateOutput(IntPtr state, FloatTensor.HType input, LongTensor.HType target, FloatTensor.HType output, long reduction, int p, FloatTensor.HType weights, double margin);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <param name="reduction"></param>
		/// <param name="p"></param>
		/// <param name="weights"></param>
		/// <param name="margin"></param>
		internal static void MultiMarginCriterion_updateOutput(FloatTensor input, LongTensor target, FloatTensor output, long reduction, int p, FloatTensor weights, double margin)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weights == null)
				throw new ArgumentNullException (nameof (weights));

			MultiMarginCriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction, p, weights.handle, margin);
		}


		[DllImport ("caffe2")]
		extern static void MultiMarginCriterion_updateGradInput(IntPtr state, FloatTensor.HType input, LongTensor.HType target, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, long reduction, int p, FloatTensor.HType weights, double margin);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="reduction"></param>
		/// <param name="p"></param>
		/// <param name="weights"></param>
		/// <param name="margin"></param>
		internal static void MultiMarginCriterion_updateGradInput(FloatTensor input, LongTensor target, FloatTensor gradOutput, FloatTensor gradInput, long reduction, int p, FloatTensor weights, double margin)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weights == null)
				throw new ArgumentNullException (nameof (weights));

			MultiMarginCriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction, p, weights.handle, margin);
		}


		[DllImport ("caffe2")]
		extern static void RReLU_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, FloatTensor.HType noise, double lower, double upper, byte train, byte inplace, IntPtr generator);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="noise"></param>
		/// <param name="lower"></param>
		/// <param name="upper"></param>
		/// <param name="train"></param>
		/// <param name="inplace"></param>
		/// <param name="generator"></param>
		internal static void RReLU_updateOutput(FloatTensor input, FloatTensor output, FloatTensor noise, double lower, double upper, bool train, bool inplace, RandomGenerator generator)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (noise == null)
				throw new ArgumentNullException (nameof (noise));

			RReLU_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, noise.handle, lower, upper, (byte)(train ? 1 : 0), (byte)(inplace ? 1 : 0), generator.handle);
		}


		[DllImport ("caffe2")]
		extern static void RReLU_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType noise, double lower, double upper, byte train, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="noise"></param>
		/// <param name="lower"></param>
		/// <param name="upper"></param>
		/// <param name="train"></param>
		/// <param name="inplace"></param>
		internal static void RReLU_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, FloatTensor noise, double lower, double upper, bool train, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (noise == null)
				throw new ArgumentNullException (nameof (noise));

			RReLU_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, noise.handle, lower, upper, (byte)(train ? 1 : 0), (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void Sigmoid_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="output">output tensor</param>
		internal static void Sigmoid_updateOutput(FloatTensor input, FloatTensor output)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			Sigmoid_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle);
		}


		[DllImport ("caffe2")]
		extern static void Sigmoid_updateGradInput(IntPtr state, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType output);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput">gradient with regards to module's output</param>
		/// <param name="gradInput">Returned value gradient with regards to input</param>
		/// <param name="output"></param>
		internal static void Sigmoid_updateGradInput(FloatTensor gradOutput, FloatTensor gradInput, FloatTensor output)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			Sigmoid_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, output.handle);
		}


		[DllImport ("caffe2")]
		extern static void SmoothL1Criterion_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType target, FloatTensor.HType output, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <param name="reduction"></param>
		internal static void SmoothL1Criterion_updateOutput(FloatTensor input, FloatTensor target, FloatTensor output, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SmoothL1Criterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void SmoothL1Criterion_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType target, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, long reduction);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="reduction"></param>
		internal static void SmoothL1Criterion_updateGradInput(FloatTensor input, FloatTensor target, FloatTensor gradOutput, FloatTensor gradInput, long reduction)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SmoothL1Criterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction);
		}


		[DllImport ("caffe2")]
		extern static void SoftPlus_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, double beta, double threshold);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="beta"></param>
		/// <param name="threshold"></param>
		internal static void SoftPlus_updateOutput(FloatTensor input, FloatTensor output, double beta, double threshold)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SoftPlus_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, beta, threshold);
		}


		[DllImport ("caffe2")]
		extern static void SoftPlus_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType output, double beta, double threshold);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="output"></param>
		/// <param name="beta"></param>
		/// <param name="threshold"></param>
		internal static void SoftPlus_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, FloatTensor output, double beta, double threshold)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SoftPlus_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, output.handle, beta, threshold);
		}


		[DllImport ("caffe2")]
		extern static void SoftShrink_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, double lambda);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="lambda"></param>
		internal static void SoftShrink_updateOutput(FloatTensor input, FloatTensor output, double lambda)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SoftShrink_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, lambda);
		}


		[DllImport ("caffe2")]
		extern static void SoftShrink_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, double lambda);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="lambda"></param>
		internal static void SoftShrink_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, double lambda)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SoftShrink_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, lambda);
		}


		[DllImport ("caffe2")]
		extern static void IndexLinear_updateOutput(IntPtr state, LongTensor.HType keys, long keysOffset, FloatTensor.HType values, LongTensor.HType sizes, LongTensor.HType cumSumSizes, FloatTensor.HType output, FloatTensor.HType weight, FloatTensor.HType bias, FloatTensor.HType normalizedValues, int train);
		/// <summary>
		/// </summary>
		/// <param name="keys"></param>
		/// <param name="keysOffset"></param>
		/// <param name="values"></param>
		/// <param name="sizes"></param>
		/// <param name="cumSumSizes"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="normalizedValues"></param>
		/// <param name="train"></param>
		internal static void IndexLinear_updateOutput(LongTensor keys, long keysOffset, FloatTensor values, LongTensor sizes, LongTensor cumSumSizes, FloatTensor output, FloatTensor weight, FloatTensor bias, FloatTensor normalizedValues, int train)
		{
			if (keys == null)
				throw new ArgumentNullException (nameof (keys));
			if (values == null)
				throw new ArgumentNullException (nameof (values));
			if (sizes == null)
				throw new ArgumentNullException (nameof (sizes));
			if (cumSumSizes == null)
				throw new ArgumentNullException (nameof (cumSumSizes));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (normalizedValues == null)
				throw new ArgumentNullException (nameof (normalizedValues));

			IndexLinear_updateOutput(IntPtr.Zero /* state */, keys.handle, keysOffset, values.handle, sizes.handle, cumSumSizes.handle, output.handle, weight.handle, bias.handle, normalizedValues.handle, train);
		}


		[DllImport ("caffe2")]
		extern static void IndexLinear_accGradParameters(IntPtr state, LongTensor.HType keys, long keysOffset, FloatTensor.HType values, LongTensor.HType sizes, LongTensor.HType cumSumSizes, FloatTensor.HType gradOutput, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType weight, FloatTensor.HType bias, FloatTensor.HType valuesBuffer, double weightDecay, double scale);
		/// <summary>
		/// </summary>
		/// <param name="keys"></param>
		/// <param name="keysOffset"></param>
		/// <param name="values"></param>
		/// <param name="sizes"></param>
		/// <param name="cumSumSizes"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="valuesBuffer"></param>
		/// <param name="weightDecay"></param>
		/// <param name="scale"></param>
		internal static void IndexLinear_accGradParameters(LongTensor keys, long keysOffset, FloatTensor values, LongTensor sizes, LongTensor cumSumSizes, FloatTensor gradOutput, FloatTensor gradWeight, FloatTensor gradBias, FloatTensor weight, FloatTensor bias, FloatTensor valuesBuffer, double weightDecay, double scale)
		{
			if (keys == null)
				throw new ArgumentNullException (nameof (keys));
			if (values == null)
				throw new ArgumentNullException (nameof (values));
			if (sizes == null)
				throw new ArgumentNullException (nameof (sizes));
			if (cumSumSizes == null)
				throw new ArgumentNullException (nameof (cumSumSizes));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (valuesBuffer == null)
				throw new ArgumentNullException (nameof (valuesBuffer));

			IndexLinear_accGradParameters(IntPtr.Zero /* state */, keys.handle, keysOffset, values.handle, sizes.handle, cumSumSizes.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, weight.handle, bias.handle, valuesBuffer.handle, weightDecay, scale);
		}


		[DllImport ("caffe2")]
		extern static void IndexLinear_accUpdateGradParameters(IntPtr state, LongTensor.HType keys, long keysOffset, FloatTensor.HType values, LongTensor.HType sizes, LongTensor.HType cumSumSizes, FloatTensor.HType gradOutput, FloatTensor.HType weight, FloatTensor.HType bias, double weightDecay, double scale);
		/// <summary>
		/// </summary>
		/// <param name="keys"></param>
		/// <param name="keysOffset"></param>
		/// <param name="values"></param>
		/// <param name="sizes"></param>
		/// <param name="cumSumSizes"></param>
		/// <param name="gradOutput"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="weightDecay"></param>
		/// <param name="scale"></param>
		internal static void IndexLinear_accUpdateGradParameters(LongTensor keys, long keysOffset, FloatTensor values, LongTensor sizes, LongTensor cumSumSizes, FloatTensor gradOutput, FloatTensor weight, FloatTensor bias, double weightDecay, double scale)
		{
			if (keys == null)
				throw new ArgumentNullException (nameof (keys));
			if (values == null)
				throw new ArgumentNullException (nameof (values));
			if (sizes == null)
				throw new ArgumentNullException (nameof (sizes));
			if (cumSumSizes == null)
				throw new ArgumentNullException (nameof (cumSumSizes));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));

			IndexLinear_accUpdateGradParameters(IntPtr.Zero /* state */, keys.handle, keysOffset, values.handle, sizes.handle, cumSumSizes.handle, gradOutput.handle, weight.handle, bias.handle, weightDecay, scale);
		}


		[DllImport ("caffe2")]
		extern static void IndexLinear_updateParameters(IntPtr state, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType weight, FloatTensor.HType bias, LongTensor.HType runningKeys, LongTensor.HType cumSumSizes, long keysOffset, double weightDecay, double learningRate);
		/// <summary>
		/// </summary>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="runningKeys"></param>
		/// <param name="cumSumSizes"></param>
		/// <param name="keysOffset"></param>
		/// <param name="weightDecay"></param>
		/// <param name="learningRate"></param>
		internal static void IndexLinear_updateParameters(FloatTensor gradWeight, FloatTensor gradBias, FloatTensor weight, FloatTensor bias, LongTensor runningKeys, LongTensor cumSumSizes, long keysOffset, double weightDecay, double learningRate)
		{
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (runningKeys == null)
				throw new ArgumentNullException (nameof (runningKeys));
			if (cumSumSizes == null)
				throw new ArgumentNullException (nameof (cumSumSizes));

			IndexLinear_updateParameters(IntPtr.Zero /* state */, gradWeight.handle, gradBias.handle, weight.handle, bias.handle, runningKeys.handle, cumSumSizes.handle, keysOffset, weightDecay, learningRate);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, FloatTensor.HType weight, FloatTensor.HType bias);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		internal static void SparseLinear_updateOutput(FloatTensor input, FloatTensor output, FloatTensor weight, FloatTensor bias)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));

			SparseLinear_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_accGradParameters(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType weight, FloatTensor.HType bias, double weightDecay, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="weightDecay"></param>
		/// <param name="scale"></param>
		internal static void SparseLinear_accGradParameters(FloatTensor input, FloatTensor gradOutput, FloatTensor gradWeight, FloatTensor gradBias, FloatTensor weight, FloatTensor bias, double weightDecay, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));

			SparseLinear_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, weight.handle, bias.handle, weightDecay, scale);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_zeroGradParameters(IntPtr state, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType lastInput);
		/// <summary>
		/// </summary>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="lastInput"></param>
		internal static void SparseLinear_zeroGradParameters(FloatTensor gradWeight, FloatTensor gradBias, FloatTensor lastInput)
		{
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (lastInput == null)
				throw new ArgumentNullException (nameof (lastInput));

			SparseLinear_zeroGradParameters(IntPtr.Zero /* state */, gradWeight.handle, gradBias.handle, lastInput.handle);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_updateParameters(IntPtr state, FloatTensor.HType weight, FloatTensor.HType bias, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType lastInput, double learningRate);
		/// <summary>
		/// </summary>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="lastInput"></param>
		/// <param name="learningRate"></param>
		internal static void SparseLinear_updateParameters(FloatTensor weight, FloatTensor bias, FloatTensor gradWeight, FloatTensor gradBias, FloatTensor lastInput, double learningRate)
		{
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (lastInput == null)
				throw new ArgumentNullException (nameof (lastInput));

			SparseLinear_updateParameters(IntPtr.Zero /* state */, weight.handle, bias.handle, gradWeight.handle, gradBias.handle, lastInput.handle, learningRate);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_legacyUpdateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, FloatTensor.HType weight, FloatTensor.HType bias);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		internal static void SparseLinear_legacyUpdateOutput(FloatTensor input, FloatTensor output, FloatTensor weight, FloatTensor bias)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));

			SparseLinear_legacyUpdateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_legacyAccGradParameters(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType weight, FloatTensor.HType bias, double weightDecay, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="weightDecay"></param>
		/// <param name="scale"></param>
		internal static void SparseLinear_legacyAccGradParameters(FloatTensor input, FloatTensor gradOutput, FloatTensor gradWeight, FloatTensor gradBias, FloatTensor weight, FloatTensor bias, double weightDecay, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));

			SparseLinear_legacyAccGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, weight.handle, bias.handle, weightDecay, scale);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_legacyZeroGradParameters(IntPtr state, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType lastInput);
		/// <summary>
		/// </summary>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="lastInput"></param>
		internal static void SparseLinear_legacyZeroGradParameters(FloatTensor gradWeight, FloatTensor gradBias, FloatTensor lastInput)
		{
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (lastInput == null)
				throw new ArgumentNullException (nameof (lastInput));

			SparseLinear_legacyZeroGradParameters(IntPtr.Zero /* state */, gradWeight.handle, gradBias.handle, lastInput.handle);
		}


		[DllImport ("caffe2")]
		extern static void SparseLinear_legacyUpdateParameters(IntPtr state, FloatTensor.HType weight, FloatTensor.HType bias, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType lastInput, double learningRate);
		/// <summary>
		/// </summary>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="lastInput"></param>
		/// <param name="learningRate"></param>
		internal static void SparseLinear_legacyUpdateParameters(FloatTensor weight, FloatTensor bias, FloatTensor gradWeight, FloatTensor gradBias, FloatTensor lastInput, double learningRate)
		{
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (lastInput == null)
				throw new ArgumentNullException (nameof (lastInput));

			SparseLinear_legacyUpdateParameters(IntPtr.Zero /* state */, weight.handle, bias.handle, gradWeight.handle, gradBias.handle, lastInput.handle, learningRate);
		}


		[DllImport ("caffe2")]
		extern static void Threshold_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, double threshold, double val, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="threshold"></param>
		/// <param name="val"></param>
		/// <param name="inplace"></param>
		internal static void Threshold_updateOutput(FloatTensor input, FloatTensor output, double threshold, double val, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			Threshold_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, threshold, val, (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void Threshold_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, double threshold, double val, byte inplace);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="threshold"></param>
		/// <param name="val"></param>
		/// <param name="inplace"></param>
		internal static void Threshold_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, double threshold, double val, bool inplace)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			Threshold_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, threshold, val, (byte)(inplace ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void TemporalRowConvolution_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, FloatTensor.HType weight, FloatTensor.HType bias, FloatTensor.HType finput, FloatTensor.HType fgradInput, int kW, int dW, int padW, byte featFirst);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kW"></param>
		/// <param name="dW"></param>
		/// <param name="padW"></param>
		/// <param name="featFirst"></param>
		internal static void TemporalRowConvolution_updateOutput(FloatTensor input, FloatTensor output, FloatTensor weight, FloatTensor bias, FloatTensor finput, FloatTensor fgradInput, int kW, int dW, int padW, bool featFirst)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			TemporalRowConvolution_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, finput.handle, fgradInput.handle, kW, dW, padW, (byte)(featFirst ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void TemporalRowConvolution_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType weight, FloatTensor.HType finput, FloatTensor.HType fgradInput, int kW, int dW, int padW, byte featFirst);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="weight"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kW"></param>
		/// <param name="dW"></param>
		/// <param name="padW"></param>
		/// <param name="featFirst"></param>
		internal static void TemporalRowConvolution_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, FloatTensor weight, FloatTensor finput, FloatTensor fgradInput, int kW, int dW, int padW, bool featFirst)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			TemporalRowConvolution_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, weight.handle, finput.handle, fgradInput.handle, kW, dW, padW, (byte)(featFirst ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void TemporalRowConvolution_accGradParameters(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType finput, FloatTensor.HType fgradInput, int kW, int dW, int padW, byte featFirst, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kW"></param>
		/// <param name="dW"></param>
		/// <param name="padW"></param>
		/// <param name="featFirst"></param>
		/// <param name="scale"></param>
		internal static void TemporalRowConvolution_accGradParameters(FloatTensor input, FloatTensor gradOutput, FloatTensor gradWeight, FloatTensor gradBias, FloatTensor finput, FloatTensor fgradInput, int kW, int dW, int padW, bool featFirst, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			TemporalRowConvolution_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, finput.handle, fgradInput.handle, kW, dW, padW, (byte)(featFirst ? 1 : 0), scale);
		}


		[DllImport ("caffe2")]
		extern static void TemporalUpSamplingNearest_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int osizeW);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeW"></param>
		internal static void TemporalUpSamplingNearest_updateOutput(FloatTensor input, FloatTensor output, int osizeW)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			TemporalUpSamplingNearest_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeW);
		}


		[DllImport ("caffe2")]
		extern static void TemporalUpSamplingNearest_updateGradInput(IntPtr state, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int isizeB, int isizeC, int isizeW, int osizeW);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="isizeB"></param>
		/// <param name="isizeC"></param>
		/// <param name="isizeW"></param>
		/// <param name="osizeW"></param>
		internal static void TemporalUpSamplingNearest_updateGradInput(FloatTensor gradOutput, FloatTensor gradInput, int isizeB, int isizeC, int isizeW, int osizeW)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			TemporalUpSamplingNearest_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, isizeB, isizeC, isizeW, osizeW);
		}


		[DllImport ("caffe2")]
		extern static void TemporalUpSamplingLinear_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int osizeW, byte align_corners);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeW"></param>
		/// <param name="align_corners"></param>
		internal static void TemporalUpSamplingLinear_updateOutput(FloatTensor input, FloatTensor output, int osizeW, bool align_corners)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			TemporalUpSamplingLinear_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeW, (byte)(align_corners ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void TemporalUpSamplingLinear_updateGradInput(IntPtr state, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int isizeB, int isizeC, int isizeW, int osizeW, byte align_corners);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="isizeB"></param>
		/// <param name="isizeC"></param>
		/// <param name="isizeW"></param>
		/// <param name="osizeW"></param>
		/// <param name="align_corners"></param>
		internal static void TemporalUpSamplingLinear_updateGradInput(FloatTensor gradOutput, FloatTensor gradInput, int isizeB, int isizeC, int isizeW, int osizeW, bool align_corners)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			TemporalUpSamplingLinear_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, isizeB, isizeC, isizeW, osizeW, (byte)(align_corners ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void BatchNormalization_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, FloatTensor.HType weight, FloatTensor.HType bias, FloatTensor.HType running_mean, FloatTensor.HType running_var, FloatTensor.HType save_mean, FloatTensor.HType save_std, byte train, double momentum, double eps);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="running_mean"></param>
		/// <param name="running_var"></param>
		/// <param name="save_mean"></param>
		/// <param name="save_std"></param>
		/// <param name="train"></param>
		/// <param name="momentum"></param>
		/// <param name="eps"></param>
		internal static void BatchNormalization_updateOutput(FloatTensor input, FloatTensor output, FloatTensor weight, FloatTensor bias, FloatTensor running_mean, FloatTensor running_var, FloatTensor save_mean, FloatTensor save_std, bool train, double momentum, double eps)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (running_mean == null)
				throw new ArgumentNullException (nameof (running_mean));
			if (running_var == null)
				throw new ArgumentNullException (nameof (running_var));
			if (save_mean == null)
				throw new ArgumentNullException (nameof (save_mean));
			if (save_std == null)
				throw new ArgumentNullException (nameof (save_std));

			BatchNormalization_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, running_mean.handle, running_var.handle, save_mean.handle, save_std.handle, (byte)(train ? 1 : 0), momentum, eps);
		}


		[DllImport ("caffe2")]
		extern static void BatchNormalization_backward(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType weight, FloatTensor.HType running_mean, FloatTensor.HType running_var, FloatTensor.HType save_mean, FloatTensor.HType save_std, byte train, double scale, double eps);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="weight"></param>
		/// <param name="running_mean"></param>
		/// <param name="running_var"></param>
		/// <param name="save_mean"></param>
		/// <param name="save_std"></param>
		/// <param name="train"></param>
		/// <param name="scale"></param>
		/// <param name="eps"></param>
		internal static void BatchNormalization_backward(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, FloatTensor gradWeight, FloatTensor gradBias, FloatTensor weight, FloatTensor running_mean, FloatTensor running_var, FloatTensor save_mean, FloatTensor save_std, bool train, double scale, double eps)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (running_mean == null)
				throw new ArgumentNullException (nameof (running_mean));
			if (running_var == null)
				throw new ArgumentNullException (nameof (running_var));
			if (save_mean == null)
				throw new ArgumentNullException (nameof (save_mean));
			if (save_std == null)
				throw new ArgumentNullException (nameof (save_std));

			BatchNormalization_backward(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, gradWeight.handle, gradBias.handle, weight.handle, running_mean.handle, running_var.handle, save_mean.handle, save_std.handle, (byte)(train ? 1 : 0), scale, eps);
		}


		[DllImport ("caffe2")]
		extern static void SpatialConvolutionMM_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, FloatTensor.HType weight, FloatTensor.HType bias, FloatTensor.HType finput, FloatTensor.HType fgradInput, int kW, int kH, int dW, int dH, int padW, int padH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		internal static void SpatialConvolutionMM_updateOutput(FloatTensor input, FloatTensor output, FloatTensor weight, FloatTensor bias, FloatTensor finput, FloatTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			SpatialConvolutionMM_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, finput.handle, fgradInput.handle, kW, kH, dW, dH, padW, padH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialConvolutionMM_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType weight, FloatTensor.HType finput, FloatTensor.HType fgradInput, int kW, int kH, int dW, int dH, int padW, int padH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="weight"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		internal static void SpatialConvolutionMM_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, FloatTensor weight, FloatTensor finput, FloatTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			SpatialConvolutionMM_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, weight.handle, finput.handle, fgradInput.handle, kW, kH, dW, dH, padW, padH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialConvolutionMM_accGradParameters(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType finput, FloatTensor.HType fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="scale"></param>
		internal static void SpatialConvolutionMM_accGradParameters(FloatTensor input, FloatTensor gradOutput, FloatTensor gradWeight, FloatTensor gradBias, FloatTensor finput, FloatTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			SpatialConvolutionMM_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, finput.handle, fgradInput.handle, kW, kH, dW, dH, padW, padH, scale);
		}


		[DllImport ("caffe2")]
		extern static void SpatialAdaptiveMaxPooling_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, LongTensor.HType indices, int osizeW, int osizeH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="indices"></param>
		/// <param name="osizeW"></param>
		/// <param name="osizeH"></param>
		internal static void SpatialAdaptiveMaxPooling_updateOutput(FloatTensor input, FloatTensor output, LongTensor indices, int osizeW, int osizeH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			SpatialAdaptiveMaxPooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, indices.handle, osizeW, osizeH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialAdaptiveMaxPooling_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, LongTensor.HType indices);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="indices"></param>
		internal static void SpatialAdaptiveMaxPooling_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, LongTensor indices)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			SpatialAdaptiveMaxPooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, indices.handle);
		}


		[DllImport ("caffe2")]
		extern static void SpatialAdaptiveAveragePooling_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int osizeW, int osizeH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeW"></param>
		/// <param name="osizeH"></param>
		internal static void SpatialAdaptiveAveragePooling_updateOutput(FloatTensor input, FloatTensor output, int osizeW, int osizeH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SpatialAdaptiveAveragePooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeW, osizeH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialAdaptiveAveragePooling_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		internal static void SpatialAdaptiveAveragePooling_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SpatialAdaptiveAveragePooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle);
		}


		[DllImport ("caffe2")]
		extern static void SpatialAveragePooling_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int kW, int kH, int dW, int dH, int padW, int padH, byte ceil_mode, byte count_include_pad);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="ceil_mode"></param>
		/// <param name="count_include_pad"></param>
		internal static void SpatialAveragePooling_updateOutput(FloatTensor input, FloatTensor output, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SpatialAveragePooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, kW, kH, dW, dH, padW, padH, (byte)(ceil_mode ? 1 : 0), (byte)(count_include_pad ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void SpatialAveragePooling_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int kW, int kH, int dW, int dH, int padW, int padH, byte ceil_mode, byte count_include_pad);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="ceil_mode"></param>
		/// <param name="count_include_pad"></param>
		internal static void SpatialAveragePooling_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SpatialAveragePooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, kW, kH, dW, dH, padW, padH, (byte)(ceil_mode ? 1 : 0), (byte)(count_include_pad ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void SpatialFractionalMaxPooling_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int outputW, int outputH, int kW, int kH, LongTensor.HType indices, FloatTensor.HType randomSamples);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="outputW"></param>
		/// <param name="outputH"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="indices"></param>
		/// <param name="randomSamples"></param>
		internal static void SpatialFractionalMaxPooling_updateOutput(FloatTensor input, FloatTensor output, int outputW, int outputH, int kW, int kH, LongTensor indices, FloatTensor randomSamples)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));
			if (randomSamples == null)
				throw new ArgumentNullException (nameof (randomSamples));

			SpatialFractionalMaxPooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, outputW, outputH, kW, kH, indices.handle, randomSamples.handle);
		}


		[DllImport ("caffe2")]
		extern static void SpatialFractionalMaxPooling_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int outputW, int outputH, int kW, int kH, LongTensor.HType indices);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="outputW"></param>
		/// <param name="outputH"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="indices"></param>
		internal static void SpatialFractionalMaxPooling_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, int outputW, int outputH, int kW, int kH, LongTensor indices)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			SpatialFractionalMaxPooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, outputW, outputH, kW, kH, indices.handle);
		}


		[DllImport ("caffe2")]
		extern static void SpatialDilatedConvolution_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, FloatTensor.HType weight, FloatTensor.HType bias, FloatTensor.HType columns, FloatTensor.HType ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="columns"></param>
		/// <param name="ones"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		internal static void SpatialDilatedConvolution_updateOutput(FloatTensor input, FloatTensor output, FloatTensor weight, FloatTensor bias, FloatTensor columns, FloatTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));
			if (ones == null)
				throw new ArgumentNullException (nameof (ones));

			SpatialDilatedConvolution_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, columns.handle, ones.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialDilatedConvolution_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType weight, FloatTensor.HType columns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="weight"></param>
		/// <param name="columns"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		internal static void SpatialDilatedConvolution_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, FloatTensor weight, FloatTensor columns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));

			SpatialDilatedConvolution_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, weight.handle, columns.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialDilatedConvolution_accGradParameters(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType columns, FloatTensor.HType ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="columns"></param>
		/// <param name="ones"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="scale"></param>
		internal static void SpatialDilatedConvolution_accGradParameters(FloatTensor input, FloatTensor gradOutput, FloatTensor gradWeight, FloatTensor gradBias, FloatTensor columns, FloatTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));
			if (ones == null)
				throw new ArgumentNullException (nameof (ones));

			SpatialDilatedConvolution_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, columns.handle, ones.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH, scale);
		}


		[DllImport ("caffe2")]
		extern static void SpatialFullDilatedConvolution_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, FloatTensor.HType weight, FloatTensor.HType bias, FloatTensor.HType columns, FloatTensor.HType ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="columns"></param>
		/// <param name="ones"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="adjW"></param>
		/// <param name="adjH"></param>
		internal static void SpatialFullDilatedConvolution_updateOutput(FloatTensor input, FloatTensor output, FloatTensor weight, FloatTensor bias, FloatTensor columns, FloatTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));
			if (ones == null)
				throw new ArgumentNullException (nameof (ones));

			SpatialFullDilatedConvolution_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, columns.handle, ones.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH, adjW, adjH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialFullDilatedConvolution_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType weight, FloatTensor.HType columns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="weight"></param>
		/// <param name="columns"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="adjW"></param>
		/// <param name="adjH"></param>
		internal static void SpatialFullDilatedConvolution_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, FloatTensor weight, FloatTensor columns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));

			SpatialFullDilatedConvolution_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, weight.handle, columns.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH, adjW, adjH);
		}


		[DllImport ("caffe2")]
		extern static void SpatialFullDilatedConvolution_accGradParameters(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType columns, FloatTensor.HType ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="columns"></param>
		/// <param name="ones"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="adjW"></param>
		/// <param name="adjH"></param>
		/// <param name="scale"></param>
		internal static void SpatialFullDilatedConvolution_accGradParameters(FloatTensor input, FloatTensor gradOutput, FloatTensor gradWeight, FloatTensor gradBias, FloatTensor columns, FloatTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));
			if (ones == null)
				throw new ArgumentNullException (nameof (ones));

			SpatialFullDilatedConvolution_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, columns.handle, ones.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH, adjW, adjH, scale);
		}


		[DllImport ("caffe2")]
		extern static void SpatialDilatedMaxPooling_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, LongTensor.HType indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, byte ceil_mode);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="indices"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="ceil_mode"></param>
		internal static void SpatialDilatedMaxPooling_updateOutput(FloatTensor input, FloatTensor output, LongTensor indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			SpatialDilatedMaxPooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, indices.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH, (byte)(ceil_mode ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void SpatialDilatedMaxPooling_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, LongTensor.HType indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, byte ceil_mode);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="indices"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="ceil_mode"></param>
		internal static void SpatialDilatedMaxPooling_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, LongTensor indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			SpatialDilatedMaxPooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, indices.handle, kW, kH, dW, dH, padW, padH, dilationW, dilationH, (byte)(ceil_mode ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void SpatialMaxUnpooling_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, LongTensor.HType indices, int owidth, int oheight);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="indices"></param>
		/// <param name="owidth"></param>
		/// <param name="oheight"></param>
		internal static void SpatialMaxUnpooling_updateOutput(FloatTensor input, FloatTensor output, LongTensor indices, int owidth, int oheight)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			SpatialMaxUnpooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, indices.handle, owidth, oheight);
		}


		[DllImport ("caffe2")]
		extern static void SpatialMaxUnpooling_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, LongTensor.HType indices, int owidth, int oheight);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="indices"></param>
		/// <param name="owidth"></param>
		/// <param name="oheight"></param>
		internal static void SpatialMaxUnpooling_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, LongTensor indices, int owidth, int oheight)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			SpatialMaxUnpooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, indices.handle, owidth, oheight);
		}


		[DllImport ("caffe2")]
		extern static void SpatialUpSamplingNearest_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int osizeH, int osizeW);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		internal static void SpatialUpSamplingNearest_updateOutput(FloatTensor input, FloatTensor output, int osizeH, int osizeW)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SpatialUpSamplingNearest_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeH, osizeW);
		}


		[DllImport ("caffe2")]
		extern static void SpatialUpSamplingNearest_updateGradInput(IntPtr state, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int isizeB, int isizeC, int isizeH, int isizeW, int osizeH, int osizeW);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="isizeB"></param>
		/// <param name="isizeC"></param>
		/// <param name="isizeH"></param>
		/// <param name="isizeW"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		internal static void SpatialUpSamplingNearest_updateGradInput(FloatTensor gradOutput, FloatTensor gradInput, int isizeB, int isizeC, int isizeH, int isizeW, int osizeH, int osizeW)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SpatialUpSamplingNearest_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, isizeB, isizeC, isizeH, isizeW, osizeH, osizeW);
		}


		[DllImport ("caffe2")]
		extern static void SpatialUpSamplingBilinear_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int osizeH, int osizeW, byte align_corners);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		/// <param name="align_corners"></param>
		internal static void SpatialUpSamplingBilinear_updateOutput(FloatTensor input, FloatTensor output, int osizeH, int osizeW, bool align_corners)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SpatialUpSamplingBilinear_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeH, osizeW, (byte)(align_corners ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void SpatialUpSamplingBilinear_updateGradInput(IntPtr state, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int isizeB, int isizeC, int isizeH, int isizeW, int osizeH, int osizeW, byte align_corners);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="isizeB"></param>
		/// <param name="isizeC"></param>
		/// <param name="isizeH"></param>
		/// <param name="isizeW"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		/// <param name="align_corners"></param>
		internal static void SpatialUpSamplingBilinear_updateGradInput(FloatTensor gradOutput, FloatTensor gradInput, int isizeB, int isizeC, int isizeH, int isizeW, int osizeH, int osizeW, bool align_corners)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SpatialUpSamplingBilinear_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, isizeB, isizeC, isizeH, isizeW, osizeH, osizeW, (byte)(align_corners ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void unfolded_acc(FloatTensor.HType finput, FloatTensor.HType input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int osizeW, int outputHeight);
		/// <summary>
		/// </summary>
		/// <param name="finput"></param>
		/// <param name="input"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="nInputPlane"></param>
		/// <param name="inputWidth"></param>
		/// <param name="inputHeight"></param>
		/// <param name="osizeW"></param>
		/// <param name="outputHeight"></param>
		internal static void unfolded_acc(FloatTensor finput, FloatTensor input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int osizeW, int outputHeight)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));

			unfolded_acc(finput.handle, input.handle, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, osizeW, outputHeight);
		}


		[DllImport ("caffe2")]
		extern static void unfolded_copy(FloatTensor.HType finput, FloatTensor.HType input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int outputWidth, int outputHeight);
		/// <summary>
		/// </summary>
		/// <param name="finput"></param>
		/// <param name="input"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="nInputPlane"></param>
		/// <param name="inputWidth"></param>
		/// <param name="inputHeight"></param>
		/// <param name="outputWidth"></param>
		/// <param name="outputHeight"></param>
		internal static void unfolded_copy(FloatTensor finput, FloatTensor input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));

			unfolded_copy(finput.handle, input.handle, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricAveragePooling_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, byte ceil_mode, byte count_include_pad);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padT"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="ceil_mode"></param>
		/// <param name="count_include_pad"></param>
		internal static void VolumetricAveragePooling_updateOutput(FloatTensor input, FloatTensor output, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, bool ceil_mode, bool count_include_pad)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			VolumetricAveragePooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, kT, kW, kH, dT, dW, dH, padT, padW, padH, (byte)(ceil_mode ? 1 : 0), (byte)(count_include_pad ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void VolumetricAveragePooling_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, byte ceil_mode, byte count_include_pad);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padT"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="ceil_mode"></param>
		/// <param name="count_include_pad"></param>
		internal static void VolumetricAveragePooling_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, bool ceil_mode, bool count_include_pad)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			VolumetricAveragePooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, kT, kW, kH, dT, dW, dH, padT, padW, padH, (byte)(ceil_mode ? 1 : 0), (byte)(count_include_pad ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void VolumetricDilatedConvolution_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, FloatTensor.HType weight, FloatTensor.HType bias, FloatTensor.HType columns, FloatTensor.HType ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="columns"></param>
		/// <param name="ones"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padT"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		internal static void VolumetricDilatedConvolution_updateOutput(FloatTensor input, FloatTensor output, FloatTensor weight, FloatTensor bias, FloatTensor columns, FloatTensor ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));
			if (ones == null)
				throw new ArgumentNullException (nameof (ones));

			VolumetricDilatedConvolution_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, columns.handle, ones.handle, kT, kW, kH, dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricDilatedConvolution_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType weight, FloatTensor.HType columns, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="weight"></param>
		/// <param name="columns"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padT"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		internal static void VolumetricDilatedConvolution_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, FloatTensor weight, FloatTensor columns, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));

			VolumetricDilatedConvolution_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, weight.handle, columns.handle, kT, kW, kH, dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricDilatedConvolution_accGradParameters(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType columns, FloatTensor.HType ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="columns"></param>
		/// <param name="ones"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="padT"></param>
		/// <param name="padW"></param>
		/// <param name="padH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="scale"></param>
		internal static void VolumetricDilatedConvolution_accGradParameters(FloatTensor input, FloatTensor gradOutput, FloatTensor gradWeight, FloatTensor gradBias, FloatTensor columns, FloatTensor ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (columns == null)
				throw new ArgumentNullException (nameof (columns));
			if (ones == null)
				throw new ArgumentNullException (nameof (ones));

			VolumetricDilatedConvolution_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, columns.handle, ones.handle, kT, kW, kH, dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH, scale);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricFullDilatedConvolution_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, FloatTensor.HType weight, FloatTensor.HType bias, FloatTensor.HType finput, FloatTensor.HType fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="aT"></param>
		/// <param name="aW"></param>
		/// <param name="aH"></param>
		internal static void VolumetricFullDilatedConvolution_updateOutput(FloatTensor input, FloatTensor output, FloatTensor weight, FloatTensor bias, FloatTensor finput, FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			VolumetricFullDilatedConvolution_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, finput.handle, fgradInput.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricFullDilatedConvolution_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType weight, FloatTensor.HType finput, FloatTensor.HType fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="weight"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="aT"></param>
		/// <param name="aW"></param>
		/// <param name="aH"></param>
		internal static void VolumetricFullDilatedConvolution_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, FloatTensor weight, FloatTensor finput, FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			VolumetricFullDilatedConvolution_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, weight.handle, finput.handle, fgradInput.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricFullDilatedConvolution_accGradParameters(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType finput, FloatTensor.HType fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="aT"></param>
		/// <param name="aW"></param>
		/// <param name="aH"></param>
		/// <param name="scale"></param>
		internal static void VolumetricFullDilatedConvolution_accGradParameters(FloatTensor input, FloatTensor gradOutput, FloatTensor gradWeight, FloatTensor gradBias, FloatTensor finput, FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			VolumetricFullDilatedConvolution_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, finput.handle, fgradInput.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH, scale);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricDilatedMaxPooling_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, LongTensor.HType indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, byte ceilMode);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="indices"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="ceilMode"></param>
		internal static void VolumetricDilatedMaxPooling_updateOutput(FloatTensor input, FloatTensor output, LongTensor indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			VolumetricDilatedMaxPooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, indices.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, (byte)(ceilMode ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void VolumetricDilatedMaxPooling_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, LongTensor.HType indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, byte ceilMode);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="indices"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		/// <param name="dilationT"></param>
		/// <param name="dilationW"></param>
		/// <param name="dilationH"></param>
		/// <param name="ceilMode"></param>
		internal static void VolumetricDilatedMaxPooling_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, LongTensor indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			VolumetricDilatedMaxPooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, indices.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, (byte)(ceilMode ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void VolumetricMaxUnpooling_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, LongTensor.HType indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="indices"></param>
		/// <param name="oT"></param>
		/// <param name="oW"></param>
		/// <param name="oH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		internal static void VolumetricMaxUnpooling_updateOutput(FloatTensor input, FloatTensor output, LongTensor indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			VolumetricMaxUnpooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, indices.handle, oT, oW, oH, dT, dW, dH, pT, pW, pH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricMaxUnpooling_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, LongTensor.HType indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="indices"></param>
		/// <param name="oT"></param>
		/// <param name="oW"></param>
		/// <param name="oH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		internal static void VolumetricMaxUnpooling_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, LongTensor indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			VolumetricMaxUnpooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, indices.handle, oT, oW, oH, dT, dW, dH, pT, pW, pH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricAdaptiveAveragePooling_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int osizeT, int osizeW, int osizeH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeT"></param>
		/// <param name="osizeW"></param>
		/// <param name="osizeH"></param>
		internal static void VolumetricAdaptiveAveragePooling_updateOutput(FloatTensor input, FloatTensor output, int osizeT, int osizeW, int osizeH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			VolumetricAdaptiveAveragePooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeT, osizeW, osizeH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricAdaptiveAveragePooling_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		internal static void VolumetricAdaptiveAveragePooling_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			VolumetricAdaptiveAveragePooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricAdaptiveMaxPooling_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, LongTensor.HType indices, int osizeT, int osizeW, int osizeH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="indices"></param>
		/// <param name="osizeT"></param>
		/// <param name="osizeW"></param>
		/// <param name="osizeH"></param>
		internal static void VolumetricAdaptiveMaxPooling_updateOutput(FloatTensor input, FloatTensor output, LongTensor indices, int osizeT, int osizeW, int osizeH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			VolumetricAdaptiveMaxPooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, indices.handle, osizeT, osizeW, osizeH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricAdaptiveMaxPooling_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, LongTensor.HType indices);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="indices"></param>
		internal static void VolumetricAdaptiveMaxPooling_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, LongTensor indices)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (indices == null)
				throw new ArgumentNullException (nameof (indices));

			VolumetricAdaptiveMaxPooling_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, indices.handle);
		}


		[DllImport ("caffe2")]
		extern static void SpatialReflectionPadding_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int pad_left, int pad_right, int pad_top, int pad_bottom);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		/// <param name="pad_top"></param>
		/// <param name="pad_bottom"></param>
		internal static void SpatialReflectionPadding_updateOutput(FloatTensor input, FloatTensor output, int pad_left, int pad_right, int pad_top, int pad_bottom)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SpatialReflectionPadding_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, pad_left, pad_right, pad_top, pad_bottom);
		}


		[DllImport ("caffe2")]
		extern static void SpatialReflectionPadding_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		/// <param name="pad_top"></param>
		/// <param name="pad_bottom"></param>
		internal static void SpatialReflectionPadding_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SpatialReflectionPadding_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, pad_left, pad_right, pad_top, pad_bottom);
		}


		[DllImport ("caffe2")]
		extern static void SpatialReplicationPadding_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int pad_left, int pad_right, int pad_top, int pad_bottom);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		/// <param name="pad_top"></param>
		/// <param name="pad_bottom"></param>
		internal static void SpatialReplicationPadding_updateOutput(FloatTensor input, FloatTensor output, int pad_left, int pad_right, int pad_top, int pad_bottom)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			SpatialReplicationPadding_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, pad_left, pad_right, pad_top, pad_bottom);
		}


		[DllImport ("caffe2")]
		extern static void SpatialReplicationPadding_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		/// <param name="pad_top"></param>
		/// <param name="pad_bottom"></param>
		internal static void SpatialReplicationPadding_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			SpatialReplicationPadding_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, pad_left, pad_right, pad_top, pad_bottom);
		}


		[DllImport ("caffe2")]
		extern static void FeatureLPPooling_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, double power, int width, int stride, byte batchMode);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="power"></param>
		/// <param name="width"></param>
		/// <param name="stride"></param>
		/// <param name="batchMode"></param>
		internal static void FeatureLPPooling_updateOutput(FloatTensor input, FloatTensor output, double power, int width, int stride, bool batchMode)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			FeatureLPPooling_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, power, width, stride, (byte)(batchMode ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void FeatureLPPooling_updateGradInput(IntPtr state, FloatTensor.HType gradOutput, FloatTensor.HType input, FloatTensor.HType output, FloatTensor.HType gradInput, double power, int width, int stride, byte batchMode);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="gradInput"></param>
		/// <param name="power"></param>
		/// <param name="width"></param>
		/// <param name="stride"></param>
		/// <param name="batchMode"></param>
		internal static void FeatureLPPooling_updateGradInput(FloatTensor gradOutput, FloatTensor input, FloatTensor output, FloatTensor gradInput, double power, int width, int stride, bool batchMode)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			FeatureLPPooling_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, input.handle, output.handle, gradInput.handle, power, width, stride, (byte)(batchMode ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void VolumetricReplicationPadding_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_back);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		/// <param name="pad_top"></param>
		/// <param name="pad_bottom"></param>
		/// <param name="pad_front"></param>
		/// <param name="pad_back"></param>
		internal static void VolumetricReplicationPadding_updateOutput(FloatTensor input, FloatTensor output, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_back)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			VolumetricReplicationPadding_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricReplicationPadding_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_back);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		/// <param name="pad_top"></param>
		/// <param name="pad_bottom"></param>
		/// <param name="pad_front"></param>
		/// <param name="pad_back"></param>
		internal static void VolumetricReplicationPadding_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_back)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			VolumetricReplicationPadding_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricUpSamplingNearest_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int osizeT, int osizeH, int osizeW);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeT"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		internal static void VolumetricUpSamplingNearest_updateOutput(FloatTensor input, FloatTensor output, int osizeT, int osizeH, int osizeW)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			VolumetricUpSamplingNearest_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeT, osizeH, osizeW);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricUpSamplingNearest_updateGradInput(IntPtr state, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int isizeB, int isizeC, int isizeT, int isizeH, int isizeW, int osizeT, int osizeH, int osizeW);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="isizeB"></param>
		/// <param name="isizeC"></param>
		/// <param name="isizeT"></param>
		/// <param name="isizeH"></param>
		/// <param name="isizeW"></param>
		/// <param name="osizeT"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		internal static void VolumetricUpSamplingNearest_updateGradInput(FloatTensor gradOutput, FloatTensor gradInput, int isizeB, int isizeC, int isizeT, int isizeH, int isizeW, int osizeT, int osizeH, int osizeW)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			VolumetricUpSamplingNearest_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, isizeB, isizeC, isizeT, isizeH, isizeW, osizeT, osizeH, osizeW);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricUpSamplingTrilinear_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int osizeT, int osizeH, int osizeW, byte align_corners);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="osizeT"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		/// <param name="align_corners"></param>
		internal static void VolumetricUpSamplingTrilinear_updateOutput(FloatTensor input, FloatTensor output, int osizeT, int osizeH, int osizeW, bool align_corners)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			VolumetricUpSamplingTrilinear_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, osizeT, osizeH, osizeW, (byte)(align_corners ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void VolumetricUpSamplingTrilinear_updateGradInput(IntPtr state, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int isizeB, int isizeC, int isizeT, int isizeH, int isizeW, int osizeT, int osizeH, int osizeW, byte align_corners);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="isizeB"></param>
		/// <param name="isizeC"></param>
		/// <param name="isizeT"></param>
		/// <param name="isizeH"></param>
		/// <param name="isizeW"></param>
		/// <param name="osizeT"></param>
		/// <param name="osizeH"></param>
		/// <param name="osizeW"></param>
		/// <param name="align_corners"></param>
		internal static void VolumetricUpSamplingTrilinear_updateGradInput(FloatTensor gradOutput, FloatTensor gradInput, int isizeB, int isizeC, int isizeT, int isizeH, int isizeW, int osizeT, int osizeH, int osizeW, bool align_corners)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			VolumetricUpSamplingTrilinear_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, isizeB, isizeC, isizeT, isizeH, isizeW, osizeT, osizeH, osizeW, (byte)(align_corners ? 1 : 0));
		}


		[DllImport ("caffe2")]
		extern static void TemporalReflectionPadding_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int pad_left, int pad_right);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		internal static void TemporalReflectionPadding_updateOutput(FloatTensor input, FloatTensor output, int pad_left, int pad_right)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			TemporalReflectionPadding_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, pad_left, pad_right);
		}


		[DllImport ("caffe2")]
		extern static void TemporalReflectionPadding_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int pad_left, int pad_right);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		internal static void TemporalReflectionPadding_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, int pad_left, int pad_right)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			TemporalReflectionPadding_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, pad_left, pad_right);
		}


		[DllImport ("caffe2")]
		extern static void TemporalReplicationPadding_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, int pad_left, int pad_right);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		internal static void TemporalReplicationPadding_updateOutput(FloatTensor input, FloatTensor output, int pad_left, int pad_right)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			TemporalReplicationPadding_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, pad_left, pad_right);
		}


		[DllImport ("caffe2")]
		extern static void TemporalReplicationPadding_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, int pad_left, int pad_right);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="pad_left"></param>
		/// <param name="pad_right"></param>
		internal static void TemporalReplicationPadding_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, int pad_left, int pad_right)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));

			TemporalReplicationPadding_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, pad_left, pad_right);
		}


		[DllImport ("caffe2")]
		extern static void Tanh_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output);
		/// <summary>
		/// </summary>
		/// <param name="input">input tensor</param>
		/// <param name="output">Returned value output tensor</param>
		internal static void Tanh_updateOutput(FloatTensor input, FloatTensor output)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			Tanh_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle);
		}


		[DllImport ("caffe2")]
		extern static void Tanh_updateGradInput(IntPtr state, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType output);
		/// <summary>
		/// </summary>
		/// <param name="gradOutput">gradient with regards to module's output</param>
		/// <param name="gradInput">Returned value gradient with regards to the input</param>
		/// <param name="output"></param>
		internal static void Tanh_updateGradInput(FloatTensor gradOutput, FloatTensor gradInput, FloatTensor output)
		{
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (output == null)
				throw new ArgumentNullException (nameof (output));

			Tanh_updateGradInput(IntPtr.Zero /* state */, gradOutput.handle, gradInput.handle, output.handle);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricConvolutionMM_updateOutput(IntPtr state, FloatTensor.HType input, FloatTensor.HType output, FloatTensor.HType weight, FloatTensor.HType bias, FloatTensor.HType finput, FloatTensor.HType fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="output"></param>
		/// <param name="weight"></param>
		/// <param name="bias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		internal static void VolumetricConvolutionMM_updateOutput(FloatTensor input, FloatTensor output, FloatTensor weight, FloatTensor bias, FloatTensor finput, FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (bias == null)
				throw new ArgumentNullException (nameof (bias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			VolumetricConvolutionMM_updateOutput(IntPtr.Zero /* state */, input.handle, output.handle, weight.handle, bias.handle, finput.handle, fgradInput.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricConvolutionMM_updateGradInput(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, FloatTensor.HType weight, FloatTensor.HType finput, FloatTensor.HType fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="weight"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		internal static void VolumetricConvolutionMM_updateGradInput(FloatTensor input, FloatTensor gradOutput, FloatTensor gradInput, FloatTensor weight, FloatTensor finput, FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weight == null)
				throw new ArgumentNullException (nameof (weight));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			VolumetricConvolutionMM_updateGradInput(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradInput.handle, weight.handle, finput.handle, fgradInput.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH);
		}


		[DllImport ("caffe2")]
		extern static void VolumetricConvolutionMM_accGradParameters(IntPtr state, FloatTensor.HType input, FloatTensor.HType gradOutput, FloatTensor.HType gradWeight, FloatTensor.HType gradBias, FloatTensor.HType finput, FloatTensor.HType fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, double scale);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradWeight"></param>
		/// <param name="gradBias"></param>
		/// <param name="finput"></param>
		/// <param name="fgradInput"></param>
		/// <param name="kT"></param>
		/// <param name="kW"></param>
		/// <param name="kH"></param>
		/// <param name="dT"></param>
		/// <param name="dW"></param>
		/// <param name="dH"></param>
		/// <param name="pT"></param>
		/// <param name="pW"></param>
		/// <param name="pH"></param>
		/// <param name="scale"></param>
		internal static void VolumetricConvolutionMM_accGradParameters(FloatTensor input, FloatTensor gradOutput, FloatTensor gradWeight, FloatTensor gradBias, FloatTensor finput, FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, double scale)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradWeight == null)
				throw new ArgumentNullException (nameof (gradWeight));
			if (gradBias == null)
				throw new ArgumentNullException (nameof (gradBias));
			if (finput == null)
				throw new ArgumentNullException (nameof (finput));
			if (fgradInput == null)
				throw new ArgumentNullException (nameof (fgradInput));

			VolumetricConvolutionMM_accGradParameters(IntPtr.Zero /* state */, input.handle, gradOutput.handle, gradWeight.handle, gradBias.handle, finput.handle, fgradInput.handle, kT, kW, kH, dT, dW, dH, pT, pW, pH, scale);
		}


		[DllImport ("caffe2")]
		extern static void SpatialClassNLLCriterion_updateOutput(IntPtr state, FloatTensor.HType input, LongTensor.HType target, FloatTensor.HType output, long reduction, FloatTensor.HType weights, FloatTensor.HType total_weight, long ignore_index);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <param name="reduction"></param>
		/// <param name="weights"></param>
		/// <param name="total_weight"></param>
		/// <param name="ignore_index"></param>
		internal static void SpatialClassNLLCriterion_updateOutput(FloatTensor input, LongTensor target, FloatTensor output, long reduction, FloatTensor weights, FloatTensor total_weight, long ignore_index)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (output == null)
				throw new ArgumentNullException (nameof (output));
			if (weights == null)
				throw new ArgumentNullException (nameof (weights));
			if (total_weight == null)
				throw new ArgumentNullException (nameof (total_weight));

			SpatialClassNLLCriterion_updateOutput(IntPtr.Zero /* state */, input.handle, target.handle, output.handle, reduction, weights.handle, total_weight.handle, ignore_index);
		}


		[DllImport ("caffe2")]
		extern static void SpatialClassNLLCriterion_updateGradInput(IntPtr state, FloatTensor.HType input, LongTensor.HType target, FloatTensor.HType gradOutput, FloatTensor.HType gradInput, long reduction, FloatTensor.HType weights, FloatTensor.HType total_weight, long ignore_index);
		/// <summary>
		/// </summary>
		/// <param name="input"></param>
		/// <param name="target"></param>
		/// <param name="gradOutput"></param>
		/// <param name="gradInput"></param>
		/// <param name="reduction"></param>
		/// <param name="weights"></param>
		/// <param name="total_weight"></param>
		/// <param name="ignore_index"></param>
		internal static void SpatialClassNLLCriterion_updateGradInput(FloatTensor input, LongTensor target, FloatTensor gradOutput, FloatTensor gradInput, long reduction, FloatTensor weights, FloatTensor total_weight, long ignore_index)
		{
			if (input == null)
				throw new ArgumentNullException (nameof (input));
			if (target == null)
				throw new ArgumentNullException (nameof (target));
			if (gradOutput == null)
				throw new ArgumentNullException (nameof (gradOutput));
			if (gradInput == null)
				throw new ArgumentNullException (nameof (gradInput));
			if (weights == null)
				throw new ArgumentNullException (nameof (weights));
			if (total_weight == null)
				throw new ArgumentNullException (nameof (total_weight));

			SpatialClassNLLCriterion_updateGradInput(IntPtr.Zero /* state */, input.handle, target.handle, gradOutput.handle, gradInput.handle, reduction, weights.handle, total_weight.handle, ignore_index);
		}


	}
}
