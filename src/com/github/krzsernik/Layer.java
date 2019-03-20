package com.github.krzsernik;

import java.util.ArrayList;
import java.util.List;

public class Layer {
    int size;
    List<Neuron> neurons = new ArrayList<>();
    Layer prev;
    Layer next;

    Layer(int s, Layer p, double lr) {
        size = s;
        prev = p;

        if(prev != null) prev.next = this;

        for(int i = 0; i < size; i++) neurons.add(new Neuron(prev, lr));
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
}