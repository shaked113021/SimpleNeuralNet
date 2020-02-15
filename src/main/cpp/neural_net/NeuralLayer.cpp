#include "neural_net/NeuralLayer.hpp"

using neural_net::NeuralLayer;

NeuralLayer::NeuralLayer(int const size)
{
  this->neurons = std::vector<Neuron>(size);
}

std::vector<double> NeuralLayer::GetOutputs(std::vector<double>& inputs)
{
  std::vector<double> output;
  for (Neuron& neuron : neurons)
  {
    output.push_back(neuron.GetOutput(inputs));
  }

  return output;
}
