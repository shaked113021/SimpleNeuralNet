#pragma once

#include <vector>
#include "Neuron.hpp"

namespace neural_net
{
    class NeuralLayer
    {
    public:
        explicit NeuralLayer(int size);
        std::vector<double> GetOutputs(std::vector<double>& inputs);
        void train(std::vector<double>& inputs, std::vector<double>& expectedOutput);
    private:
        std::vector<Neuron> neurons;
    };
}