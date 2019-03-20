package com.github.krzsernik;

import java.util.ArrayList;
import java.util.List;

public class Layer {
    int size;
    List<Neuron> neurons = new ArrayList<>();
    Layer prev;
    Layer next;

    Layer(int s, Layer p, double lr) {
        this.size = s;
        this.prev = p;

        if(this.prev != null) this.prev.next = this;

        for(int i = 0; i < this.size; i++) this.neurons.add(new Neuron(this.prev, lr));
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

    void propagate(List<Double> target) {
        for(int i = this.size - 1; i >= 0; i--) {
            this.neurons.get(i).propagate(target.get(i));
        }
    }
    void propagate() {
        for(int i = this.size - 1; i >= 0; i--) {
            this.neurons.get(i).propagate();
        }
    }
}