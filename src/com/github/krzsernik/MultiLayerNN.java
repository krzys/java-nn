package com.github.krzsernik;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

// TODO: training method
// TODO: retraining method
// TODO: weights adjusting method
// TODO: cost calculation method
// TODO: predict method
// TODO: something I forgot to mention

public class MultiLayerNN {
    private List<Vector<Vector<Double>>> m_weights = new ArrayList<>();
    private List<Integer> m_hiddenLayers = new ArrayList<>();
    private double m_learningRate = 0.01;

    public MultiLayerNN(int inputs, List<Integer> hiddenLayers, int outputs) {
        m_hiddenLayers = hiddenLayers;

        int nextLayerNeurons = outputs;
        if(hiddenLayers.size() > 0) {
            nextLayerNeurons = hiddenLayers.get(0);
        }

        Vector<Vector<Double>> matrix = new Vector<>();
        for(int input = 0; input < inputs; input++) {
            Vector<Double> row = new Vector<>();
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
                Vector<Double> row = new Vector<>();

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
}
