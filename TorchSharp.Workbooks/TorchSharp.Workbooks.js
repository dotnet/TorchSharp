console.log("hello from some third party code => %O", xamarin.interactive)

var TensorRenderer = (function () {
  function TensorRenderer () {
  }

  TensorRenderer.prototype.cssClass = "renderer-third-party-person";

  TensorRenderer.prototype.getRepresentations = function () {
    return [
      { shortDisplayName: "Tensor" }
    ]
  };

  TensorRenderer.prototype.bind = function (renderState) {
    console.log("TensorRenderer: bind: %O", renderState)
    this.renderState = renderState;
  };

  TensorRenderer.prototype.render = function (target) {
    console.log("TensorRenderer: render %O to %O", this.renderState, target)
    var elem = document.createElement("div");
    elem.innerHTML = "<strong>Tensor: <em>" + this.renderState.source.Name + "</em></strong>";
    target.inlineTarget.appendChild(elem);
  }

  return TensorRenderer;
})();

xamarin.interactive.RendererRegistry.registerRenderer(
  function (source) {
    if (source.$type === "TorchSharp.FloatTensor")
      return new TensorRenderer;
  }
);