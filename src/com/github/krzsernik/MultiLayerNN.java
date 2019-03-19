package com.github.krzsernik;

import java.util.ArrayList;
import java.util.List;

// TODO: weights adjusting method
// TODO: something I forgot to mention

public class MultiLayerNN {
    private class Data {
        public List<Double> in;
        public List<Double> out;
    }
    private List<Data> m_data = new ArrayList<>();

    private List<List<List<Double>>> m_weights = new ArrayList<>();
    private List<Integer> m_hiddenLayers = new ArrayList<>();
    private int m_inputs;
    private int m_outputs;
    private double m_learningRate = 0.01;
    private double m_errorRate = 0.05;

    public MultiLayerNN(int inputs, List<Integer> hiddenLayers, int outputs) {
        m_inputs = inputs;
        m_hiddenLayers = hiddenLayers;
        m_outputs = outputs;

        int nextLayerNeurons = outputs;
        if(hiddenLayers.size() > 0) {
            nextLayerNeurons = hiddenLayers.get(0);
        }

        List<List<Double>> matrix = new ArrayList<>();
        for(int input = 0; input < inputs; input++) {
            List<Double> row = new ArrayList<>();
            for(int nextLayerNeuron = 0; nextLayerNeuron < nextLayerNeurons; nextLayerNeuron++) {
                row.add(Math.random());
            }
            matrix.add(row);
        }
        m_weights.add(matrix);

        for(int layer = 0, size = m_hiddenLayers.size(); layer < size; layer++) {
            if(size - layer == 1) {
                nextLayerNeurons = outputs;
            } else {
                nextLayerNeurons = m_hiddenLayers.get(layer + 1);
            }

            matrix.clear();
            for(int currentLayerNeuron = 0; currentLayerNeuron < m_hiddenLayers.get(layer); currentLayerNeuron++) {
                List<Double> row = new ArrayList<>();

                for(int nextLayerNeuron = 0; nextLayerNeuron < nextLayerNeurons; nextLayerNeuron++) {
                    row.add(Math.random());
                }
                matrix.add(row);
            }

            m_weights.add(matrix);
        }
    }
    public void setLearningRate(double learningRate) {
        m_learningRate = learningRate;
    }
    public void setErrorRate(double errorRate) {
        m_errorRate = errorRate;
    }

    public boolean train(List<Double> inputs, List<Double> expected) {
        List<Double> result = predict(inputs);
        m_data.add(new Data(){{
            this.in = inputs;
            this.out = expected;
        }});

        double error = 0;
        for(int i = 0; i < result.size(); i++) {
            error += Math.pow(result.get(i) - expected.get(i), 2);
        }
        error /= result.size();

        if(error < m_errorRate) return true;
        else {
            adjust(result, expected);

            return false;
        }
    }
    public boolean retrain() {
        int size = m_data.size();
        boolean success = true;

        for(int i = 0; i < size; i++) {
            Data training = m_data.get(0);
            m_data.remove(0);

            success = train(training.in, training.out) && success;
        }

        return success;
    }

    public List<Double> predict(List<Double> inputs) {
        for(int layer = 0; layer <= m_hiddenLayers.size(); layer++) {
            inputs = getWeightedSum(layer, inputs, true);
        }

        return inputs;
    }

    private List<Double> getWeightedSum(int layer, List<Double> inputs, boolean sigmoid) {
        List<Double> result = new ArrayList<>();
        List<List<Double>> matrix = m_weights.get(layer);

        for(int row = 0; row < matrix.size(); row++) {
            double neuronActivation = 0;

            for(int neuron = 0; neuron < matrix.get(row).size(); neuron++) {
                neuronActivation += matrix.get(row).get(neuron) * inputs.get(row);
            }

            if(sigmoid) {
                neuronActivation = sigmoid(neuronActivation);
            }

            result.add(neuronActivation);
        }

        return result;
    }

    private double sigmoid(double v) {
        return 1 / (1 + Math.pow(Math.E, -v));
    }

    private void adjust(List<Double> inputs, List<Double> expected) {
        List<Double> result = inputs;
        List<List<Double>> outputs = new ArrayList<>();

        for(int layer = 0; layer < m_hiddenLayers.size(); layer++) {
            result = getWeightedSum(layer, result, true);
            outputs.add(result);
        }

        for(int layer = m_hiddenLayers.size() - 1; layer >= 0; layer++) {
            for(int neuron = 0; neuron < m_hiddenLayers.get(layer); neuron++) {
                for(int weight = 0; weight < m_weights.get(layer).get(neuron).size(); weight++) {
                    double weightValue = m_weights.get(layer).get(neuron).get(weight);
                    weightValue = cost(outputs.get(layer).get(neuron), expected.get(neuron));

                    m_weights.get(layer).get(neuron).set(weight, weightValue);
                }
            }
        }
    }

    private double cost(double y, double d) {
        return Math.pow(d - y, 2);
    }
}
