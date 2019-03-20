package com.github.krzsernik;

import java.util.ArrayList;
import java.util.List;

public class Neuron {
    List<Connection> connections = new ArrayList<>();
    double activation = 0;
    double derivative = 0;
    double bias;
    double learningRate;

    Neuron(Layer prev, double lr) {
        this.learningRate = lr;
        this.bias = Math.random() * 0.2 - 0.1;

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
        weightedSum += this.bias;

        this.activation = this.sigmoid(weightedSum);
        this.derivative = this.derivate(this.activation);

        return this.activation;
    }

    private double sigmoid(double v) {
        return 1 / (1 + Math.pow(Math.E, -v));
    }
    private double derivate(double y) {
        return y * (1 - y);
    }

    void AddConnection(Neuron to) {
        connections.add(new Connection(this, to));
    }
    void AddConnection(Neuron from, Neuron to) {
        connections.add(new Connection(from, to));
    }
}