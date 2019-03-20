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

    private double m_learningRate = 0.01;
    private double m_errorRate = 0.05;

    public MultiLayerNN(int inputs, List<Integer> hiddenLayers, int outputs) {
        m_inputLayer = new Layer(inputs, null, m_learningRate);

        for(int neurons : hiddenLayers) {
            if(m_hiddenLayers.size() > 0) {
                Layer prev = m_hiddenLayers.get(m_hiddenLayers.size() - 1);
                m_hiddenLayers.add(new Layer(neurons, prev, m_learningRate));
            } else {
                m_hiddenLayers.add(new Layer(neurons, m_inputLayer, m_learningRate));
            }
        }

        Layer lastHiddenLayer = m_hiddenLayers.get(m_hiddenLayers.size() - 1);
        m_outputLayer = new Layer(outputs, lastHiddenLayer, m_learningRate);
    }
    public void setLearningRate(double learningRate) {
        m_learningRate = learningRate;

        Layer layer = m_inputLayer;
        while(layer != null) {
            for(Neuron n : layer.neurons) {
                n.learningRate = learningRate;
            }
            layer = layer.next;
        }
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

        System.out.println(activations);

        return false;
    }

    private double activate(double v) {
        return 1 / (1 + Math.pow(Math.E, -v));
    }
    private double derivative(double v) {
        return v * (1 - v);
    }
}
