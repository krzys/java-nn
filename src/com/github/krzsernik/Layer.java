package com.github.krzsernik;

import java.util.ArrayList;
import java.util.List;

public class Layer {
    int size;
    List<Neuron> neurons = new ArrayList<>();
    Layer prev;
    Layer next;

    Layer(int s, Layer p, Layer n) {
        size = s;
        prev = p;
        next = n;

        if(prev != null) prev.next = this;
        if(next != null) next.prev = this;

        for(int i = 0; i < size; i++) neurons.add(new Neuron(prev));
    }
}