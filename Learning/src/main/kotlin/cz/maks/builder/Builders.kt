package cz.maks.builder

import cz.maks.init
import cz.maks.model.*
import cz.maks.strategies.*

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */

class DenseNetworkBuilder(
        private val inputCount: Int,
        private val activationFunction: ActivationFunction,
        private val hiddenLayerNeurons: MutableList<Int> = ArrayList(),
        private var lossFunction: LossFunction = Loss.difference(),
        private var weightInitialisationFunction: WeightInitialisationFunction = WeightInitialisation.xavierNormal(),
        private var biasInitialisationFunction: BiasInitialisationFunction = BiasInitialisation.zeros()
) {
    companion object {
        private fun generateInputs(inputCount: Int): Array<Input> {
            val inputs = ArrayList<Input>()
            for (i in 0 until inputCount) {
                inputs.add(Input())
            }
            return inputs.toTypedArray()
        }

        private fun generateOutputs(outputCount: Int, activationFunction: ActivationFunction): Array<Neuron> {
            val outputs = ArrayList<Neuron>()
            for (i in 0 until outputCount) {
                outputs.add(Neuron(
                        activationFunction = activationFunction,
                        name = "output$i"
                ))
            }
            return outputs.toTypedArray()
        }

        private fun generateHiddenNeurons(outputCount: Int, order: Int, activationFunction: ActivationFunction): Array<Neuron> {
            val neurons = ArrayList<Neuron>()
            for (i in 0 until outputCount) {
                neurons.add(Neuron(
                        activationFunction = activationFunction,
                        name = "hidden $order-$i"
                ))
            }
            return neurons.toTypedArray()
        }

        // Extension
        private fun NeuralNetwork.wireUp(inputs: Array<out ConnectionInput>, outputs: Array<Neuron>) {
            inputs.forEach {
                val input = it
                outputs.forEach {
                    connect(input, it)
                }
            }
        }
    }

    fun addHiddenLayer(neuronCount: Int): DenseNetworkBuilder {
        hiddenLayerNeurons.add(neuronCount)
        return this
    }

    fun lossFunction(lossFunction: LossFunction): DenseNetworkBuilder {
        this.lossFunction = lossFunction
        return this
    }

    fun biasInitializationFunction(biasInitialisationFunction: BiasInitialisationFunction): DenseNetworkBuilder {
        this.biasInitialisationFunction = biasInitialisationFunction
        return this
    }

    fun weightInitialisationFunction(weightInitialisationFunction: WeightInitialisationFunction): DenseNetworkBuilder {
        this.weightInitialisationFunction = weightInitialisationFunction
        return this
    }

    fun build(outputCount: Int): NeuralNetwork {
        val network = generateNetwork(outputCount)

        val neurons = ArrayList<Array<Neuron>>()
        for (hiddenLayer in network.hiddenLayers) {
            neurons.add(hiddenLayer.neurons)
        }
        neurons.add(network.outputLayer.neurons)

        if (neurons.size > 1) {
            for (i in 0 until neurons.size - 1) {
                network.wireUp(neurons[i], neurons[i + 1])
            }
        }

        network.wireUp(network.inputLayer.inputs, neurons[0])
        network.init()
        return network
    }

    private fun generateNetwork(outputCount: Int): NeuralNetwork {
        val inputLayer = InputLayer(generateInputs(inputCount))
        val outputLayer = OutputLayer(generateOutputs(outputCount, activationFunction))
        val network = NeuralNetwork(
                inputLayer = inputLayer,
                outputLayer = outputLayer,
                lossFunction = lossFunction,
                weightInitialisationFunction = weightInitialisationFunction,
                biasInitialisationFunction = biasInitialisationFunction
        )

        for (ind in 0 until hiddenLayerNeurons.size) {
            val hiddenLayer = generateHiddenNeurons(hiddenLayerNeurons[ind], ind, activationFunction)
            network.addHiddenLayer(hiddenLayer)
        }

        return network
    }

}