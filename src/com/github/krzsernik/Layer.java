package com.github.krzsernik;

import java.util.ArrayList;
import java.util.List;

public class Layer {
    int size;
    List<Neuron> neurons = new ArrayList<>();
    Layer prev;
    Layer next;

    Layer(int s, Layer p) {
        this.size = s;
        this.prev = p;

        if(this.prev != null) this.prev.next = this;

        for(int i = 0; i < this.size; i++) this.neurons.add(new Neuron(this.prev));
    }

    void activate(List<Double> inputs) {
        int index = 0;
        for (Neuron n : neurons) {
            n.activation = inputs.get(index++);
        }
    }
    List<Double> activate() {
        List<Double> result = new ArrayList<>();

        for(Neuron n : neurons) {
            result.add(n.activate());
        }

        return result;
    }

    void propagate(List<Double> target, double learningRate) {
        List<Double> output = this.activate();

        for(int i = 0; i < this.size; i++) {
            Neuron currentNeuron = this.neurons.get(i);
            double error = target.get(i) - output.get(i);
            currentNeuron.delta = currentNeuron.derivative * error;

            for(Connection input : currentNeuron.connections) {
                if(input.to == currentNeuron) {
                    input.weight += learningRate * currentNeuron.delta * input.from.activation;
                }
            }
        }
    }
    void propagate(double learningRate) {
        for(Neuron currentNeuron : this.neurons) {
            double error = 0.0;

            for(Connection connection : currentNeuron.connections) {
                if(connection.from == currentNeuron) {
                    error += connection.to.delta * connection.weight * currentNeuron.derivative;
                }
            }

            currentNeuron.delta = error;

            for(Connection input : currentNeuron.connections) {
                if(input.to == currentNeuron) {
                    input.weight += learningRate * currentNeuron.delta * input.from.activation;
                }
            }
        }
    }
}