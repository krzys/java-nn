package com.github.krzsernik;

import java.util.ArrayList;
import java.util.List;

public class MultiLayerNN {
    private class Data {
        List<Double> in;
        List<Double> out;
    }
    private List<Data> m_data = new ArrayList<>();

    private Layer m_inputLayer;
    private List<Layer> m_hiddenLayers = new ArrayList<>();
    private Layer m_outputLayer;

    private double m_learningRate = 0.1;
    private double m_errorRate = 0.05;

    public MultiLayerNN(int inputs, List<Integer> hiddenLayers, int outputs) {
        m_inputLayer = new Layer(inputs, null);

        for(int neurons : hiddenLayers) {
            if(m_hiddenLayers.size() > 0) {
                Layer prev = m_hiddenLayers.get(m_hiddenLayers.size() - 1);
                m_hiddenLayers.add(new Layer(neurons, prev));
            } else {
                m_hiddenLayers.add(new Layer(neurons, m_inputLayer));
            }
        }

        Layer lastHiddenLayer = m_hiddenLayers.size() > 0 ?
                                    m_hiddenLayers.get(m_hiddenLayers.size() - 1) :
                                    m_inputLayer;
        m_outputLayer = new Layer(outputs, lastHiddenLayer);
    }
    public void setLearningRate(double learningRate) {
        m_learningRate = learningRate;
    }
    public void setErrorRate(double errorRate) {
        m_errorRate = errorRate;
    }
    public boolean train(List<Double> inputs, List<Double> expected) {
        List<List<Double>> activations = new ArrayList<>();

        m_inputLayer.activate(inputs);

        Layer layer = m_hiddenLayers.size() > 0 ? m_hiddenLayers.get(0) : m_outputLayer;
        while(layer != null) {
            activations.add(layer.activate());

            layer = layer.next;
        }

        double error = calcError(activations.get(activations.size() - 1), expected);

        propagate(expected);

        if(error < m_errorRate) return true;

        m_data.add(new Data(){{
            in = inputs;
            out = expected;
        }});
        return false;
    }
    public boolean retrain() {
        boolean success = true;

        for(int i = 0; i < m_data.size(); i++) {
            Data training = m_data.get(0);
            m_data.remove(0);

            success = train(training.in, training.out) && success;
        }

        return success;
    }
    public List<Double> predict(List<Double> inputs) {
        List<Double> result = new ArrayList<>();
        m_inputLayer.activate(inputs);

        Layer layer = m_hiddenLayers.size() > 0 ? m_hiddenLayers.get(0) : m_outputLayer;
        while(layer != null) {
            result = layer.activate();

            layer = layer.next;
        }

        return result;
    }

    private void propagate(List<Double> expected) {
        m_outputLayer.propagate(expected, m_learningRate);

        for(int hidden = m_hiddenLayers.size() - 1; hidden >= 0; hidden--) {
            m_hiddenLayers.get(hidden).propagate(m_learningRate);
        }
        m_inputLayer.propagate(m_learningRate);
    }

    private double calcError(List<Double> out, List<Double> expected) {
        double result = 0;

        for(int i = 0; i < out.size(); i++) {
            result += 0.5 * Math.pow(expected.get(i) - out.get(i), 2);
        }

        return result;
    }
}
