using System;
using System.Collections.Generic;
using Xamarin.Interactive;
using Xamarin.Interactive.Representations;
using Xamarin.Interactive.Logging;
using Xamarin.Interactive.Serialization;

[assembly: AgentIntegration (typeof (TorchSharp.Workbooks.TensorIntegration))]

namespace TorchSharp.Workbooks {
	public class TensorRepresentationProvider : RepresentationProvider {
		public override IEnumerable<object> ProvideRepresentations (object obj)
		{
			if (obj is FloatTensor ft)
				yield return new FloatTensorRepresentation (ft);
	    		else
				yield return null;
		}
	}

	public class TensorIntegration : IAgentIntegration {
		public TensorIntegration ()
		{
		}

		public void IntegrateWith (IAgent agent)
		{
			Log.Debug ("Tensor", "Loading");

			agent.RepresentationManager.AddProvider (new TensorRepresentationProvider ());
		}
	}

	public class FloatTensorRepresentation : ISerializableObject
	{
		readonly FloatTensor tensor;

		public int Dimensions => tensor.Dimensions;

		public FloatTensorRepresentation (FloatTensor tensor)
		{
			this.tensor = tensor ?? throw new ArgumentNullException (nameof (tensor));
		}

		void ISerializableObject.Serialize (ObjectSerializer serializer)
		{
			serializer.Property (nameof (Dimensions), Dimensions);
		}
	}
}
