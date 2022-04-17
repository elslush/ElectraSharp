using Tensorflow;

namespace Electra.Model.Activations;

public delegate Tensor ActivationFunction(Tensor x, string? name = null);
