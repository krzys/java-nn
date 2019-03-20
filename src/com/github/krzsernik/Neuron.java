package com.github.krzsernik;

import java.util.ArrayList;
import java.util.List;

public class Neuron {
    Neuron(Layer prev) {
        if(prev == null) return;

        for(Neuron n : prev.neurons) {
            n.AddConnection(this);
        }
    }

    Neuron() {}

    void AddConnection(Neuron to) {
        connections.add(new Connection(this, to));
    }

    List<Connection> connections = new ArrayList<>();
}