#include <cstdlib>
#include <tgmath.h>
#include "neural_net/Neuron.hpp"

using neural_net::Neuron;

Neuron::Neuron(int const inputs, double const learnRate) : m_learnRate(learnRate)
{
  this->m_weights = std::vector<double>(inputs);

  for(double &w : m_weights)
  {
    w = (double)(rand() % 10000 + 1) / 10000 - 0.5;
  }
  m_bias = (double)(rand() % 10000 + 1) / 10000 - 0.5;
}

double Neuron::sigmoid(double const x)
{
  return 1 / (1 + exp(-x));
}

double Neuron::sigmoidPrime(double const x)
{
  return exp(-x) / pow((1 + exp(-x)), 2);
}

double Neuron::GetOutput(std::vector<double> &inputs) const
{
  double sum = m_bias;

  for (int i = 0; i < inputs.size(); ++i)
  {
    sum += inputs[i] * m_weights[i];
  }
  return sigmoid(sum);
}

double Neuron::squaredError(double const a, double const b)
{
  return pow((a - b), 2) / 2;
}

double Neuron:: squaredErrorPrime(double const a, double const b)
{
  return a - b;
}

void neural_net::Neuron::train(double const predictionCostGradient, double const predictionGradient, std::vector<double> &inputs)
{
  double biasCostGradient = predictionCostGradient * predictionGradient;
  std::vector<double> weightsCostGradient(m_weights.size());

  for (int i = 0; i < inputs.size(); ++i)
  {
    weightsCostGradient[i] = inputs[i] * biasCostGradient;
  }

  m_bias -= m_learnRate * biasCostGradient;

  for (int i = 0; i < m_weights.size(); ++i)
  {
    m_weights[i] -= m_learnRate * weightsCostGradient[i];
  }
}

double neural_net::Neuron::logit(double x)
{
  return log(x / (1 - x));
}
