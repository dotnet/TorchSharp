using System;
using System.Collections.Generic;
using Xamarin.Interactive;
using Xamarin.Interactive.Representations;

[assembly: AgentIntegration (typeof (TorchSharp.Workbooks.TensorIntegration))]

namespace TorchSharp.Workbooks {
	public class TensorRepresentationProvider : RepresentationProvider {
		public override IEnumerable<object> ProvideRepresentations (object obj)
		{
			return base.ProvideRepresentations (obj);
		}
	}

	public class TensorIntegration : IAgentIntegration {
		public TensorIntegration ()
		{
		}

		public void IntegrateWith (IAgent agent)
		{
			agent.RepresentationManager.AddProvider (new TensorRepresentationProvider ());
		}
	}
}
