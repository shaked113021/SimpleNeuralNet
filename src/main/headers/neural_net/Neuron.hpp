#pragma once

#include <vector>

namespace neural_net
{
    class Neuron
    {
    public:
        Neuron(int inputs, double learnRate);
        double GetOutput(std::vector<double>& inputs) const;
        void train(double predictionCostGradient, double predictionGradient, std::vector<double>& inputs);
        static double sigmoidPrime(double x);
        static double squaredErrorPrime(double a, double b);
        static double logit(double x);
    private:
        static double sigmoid(double x);
        static double squaredError(double a, double b);
        std::vector<double> m_weights;
        double m_learnRate;
        double m_bias;
    };
}