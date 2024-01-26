using System;
using System.Linq;
using System.Collections.Generic;
using static TorchSharp.torch.autograd;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class autograd
        {
            public static partial class functional
            {
                /// <summary>
                /// Preprocesses the inputs to make sure they require gradient.
                /// </summary>
                /// <param name="need_graph">Specifies if we internally want gradients to flow back to the Tensors in the result.</param>
                /// <param name="inputs">The tensors to preprocess.</param>
                /// <returns>The same tensors as inputs, but requiring grad.</returns>
                private static Tensor[] _grad_preprocess(bool need_graph=true, params Tensor[] inputs)
                {
                    List<Tensor> result = new List<Tensor>();
                    foreach (var input in inputs)
                    {
                        result.Add(input.detach().requires_grad_(need_graph));
                    }
                    return result.ToArray();
                }

                /// <summary>
                /// Wrapper function around torch.autograd.grad.
                /// </summary>
                /// <param name="output">The output tensor to compute gradients for.</param>
                /// <param name="retain_graph">Whether the gradient computation graph should be kept.</param>
                /// <param name="inputs">The input tensors at which to evaluate the gradients.</param>
                /// <returns>A collection of gradient tensors for the output with respect to each input.</returns>
                private static IEnumerable<Tensor> _autograd_grad(Tensor output, bool retain_graph=false, params Tensor[] inputs)
                {
                    var outputs = new List<Tensor>
                    {
                        output
                    };
                    var inputList = new List<torch.Tensor>(inputs);
                    return torch.autograd.grad(outputs, inputList, retain_graph: retain_graph, allow_unused: true);
                }

                /// <summary>
                /// Computes the jacobian of a given function.
                /// 
                /// Example:
                /// torch.Tensor x1 = torch.tensor(new double[,] { { 1.0, 3.0 }, { 2.0, 4.0 } }, requires_grad: true);
                /// torch.Tensor intercepts = torch.tensor(new double[] { 1.0, 5.0 }, requires_grad: true);
                /// 
                /// torch.Tensor[] jacFunc(torch.Tensor[] inputs)
                /// {
                ///     return new torch.Tensor[] { torch.einsum("ij,j->i", inputs[0], inputs[1]) };
                /// }
                ///
                /// var jacobian = torch.autograd.functional.jacobian(jacFunc, x1, intercepts);
                ///
                /// jacobian[0] should be:
                /// [[[1, 5],
                ///   [0, 0]],
                ///  [[0, 0],
                ///   [1, 5]]]
                /// And jacobian[1]:
                /// [[1, 3],
                ///  [2, 4]]
                ///  
                /// </summary>
                /// <param name="function">The function to compute the jacobian for. It may have multiple inputs and multiple outputs.</param>
                /// <param name="inputs">The values for the inputs at which to compute the jacobian.</param>
                /// <returns>A list of tensors, one for each output value of each output. For example, a single output with two elements yields two jacobian tensors.</returns>
                public static IEnumerable<Tensor> jacobian(Func<Tensor[], Tensor[]> function, params Tensor[] inputs)
                {
                    List<Tensor> jacobians = new List<Tensor>();
                    using (var d0 = torch.NewDisposeScope()) {
                        using (enable_grad()) {
                            inputs = _grad_preprocess(need_graph: true, inputs);
                            var output = function(inputs);
                            for (int i = 0; i < output.Length; i++) // i is output variable
                            {
                                var jacobian_i = new List<Tensor>[inputs.Length]; // going to have as many elements as there are input variables
                                for (int k = 0; k < inputs.Length; k++)
                                    jacobian_i[k] = new List<Tensor>();

                                for (int j = 0; j < output[i].NumberOfElements; j++) {
                                    var vj = _autograd_grad(output[i].reshape(-1)[j], retain_graph: true, inputs);
                                    foreach ((List<Tensor> jac_i_el, Tensor vj_el, Tensor inp_el) combi in jacobian_i.Zip(vj, inputs)) {
                                        if (combi.vj_el.IsInvalid) // how to check for null tensors?
                                            combi.jac_i_el.Add(zeros_like(combi.inp_el));
                                        else
                                            combi.jac_i_el.Add(combi.vj_el);
                                    }
                                }
                                foreach ((List<Tensor> jac_i_el, Tensor inp_el) row in jacobian_i.Zip(inputs)) {
                                    jacobians.Add(
                                        stack(row.jac_i_el, dim: 0)
                                                .view(output[i].size().Concat(row.inp_el.size()).ToArray())
                                                .MoveToOuterDisposeScope() // this is from the _grad_postprocess because create_graph is always false in this implementation
                                    );
                                }
                            }
                        }
                        return jacobians;
                    }
                }

                /// <summary>
                /// Computes the jacobian of a given function with one input and one output.
                /// </summary>
                /// <param name="function">The function to compute the jacobian for. It has a single input and single output.</param>
                /// <param name="inputs">The value for the input at which to compute the jacobian.</param>
                /// <returns>A single tensor containing the jacobian.</returns>
                public static Tensor jacobian(Func<Tensor, Tensor> function, Tensor inputs)
                {
                    var wrapper = new Func<Tensor[], Tensor[]>((Tensor[] x) => {
                        return new Tensor[] { function(x.Single()) };
                    });
                    return jacobian(wrapper, new Tensor[] { inputs }).Single();
                }

                /// <summary>
                /// Computes the jacobian of a given function with one input and multiple outputs.
                /// </summary>
                /// <param name="function">The function to compute the jacobian for. It has a single input and multiple outputs.</param>
                /// <param name="inputs">The value for the input at which to compute the jacobian.</param>
                /// <returns>A list of tensors containing the jacobian for each output.</returns>
                public static IEnumerable<Tensor> jacobian(Func<Tensor, Tensor[]> function, Tensor inputs)
                {
                    var wrapper = new Func<Tensor[], Tensor[]>((Tensor[] x) => {
                        return function(x.Single());
                    });
                    return jacobian(wrapper, new Tensor[] { inputs });
                }

                /// <summary>
                /// Computes the jacobian of a given function with multiple inputs and one output.
                /// </summary>
                /// <param name="function">The function to compute the jacobian for. It has multiple inputs and single output.</param>
                /// <param name="inputs">The values for the inputs at which to compute the jacobian.</param>
                /// <returns>A list of tensors containing the jacobian for the output with respect to each input.</returns>
                public static IEnumerable<Tensor> jacobian(Func<Tensor[], Tensor> function, params Tensor[] inputs)
                {
                    var wrapper = new Func<Tensor[], Tensor[]>((Tensor[] x) => {
                        return new Tensor[] { function(x) };
                    });
                    return jacobian(wrapper, inputs);
                }
            }
        }
    }
}