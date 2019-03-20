package com.github.krzsernik;

public class Connection {
    Connection(Neuron f, Neuron t) {
        from = f;
        to = t;
        weight = Math.random();
    }

    Neuron from;
    Neuron to;
    double weight;
}