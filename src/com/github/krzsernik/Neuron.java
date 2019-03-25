package com.github.krzsernik;

import java.util.ArrayList;
import java.util.List;

public class Neuron {
    List<Connection> connections = new ArrayList<>();
    double activation = 0;
    double derivative = 0;
    double delta = 0;

    Neuron(Layer prev) {
        if(prev == null) return;

        for(Neuron n : prev.neurons) {
            n.AddConnection(this);
            this.AddConnection(n, this);
        }
    }

    double activate() {
        double weightedSum = 0;

        for(Connection c : this.connections) {
            if(c.to == this) {
                weightedSum += c.weight * c.from.activation;
            }
        }

        this.activation = this.sigmoid(weightedSum);
        this.derivative = this.derivate(this.activation);

        return this.activation;
    }

    private double sigmoid(double v) {
        return 1 / (1 + Math.pow(Math.E, -v));
    }
    private double derivate(double y) {
        return y - Math.pow(y, 2); // y * (1 - y)
    }

    void AddConnection(Neuron to) {
        this.connections.add(new Connection(this, to));
    }
    void AddConnection(Neuron from, Neuron to) {
        this.connections.add(new Connection(from, to));
    }
}