package cz.maks.model

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */

class DenseNetworkBuilder(
        private val inputCount: Int,
        private val triggerFunction: TriggerFunction,
        private val hiddenLayerNeurons: MutableList<Int> = ArrayList()
) {
    companion object {
        fun generateInputs(inputCount: Int): List<Input> {
            val inputs = ArrayList<Input>()
            for (i in 0 until inputCount) {
                inputs.add(Input())
            }
            return inputs
        }

        fun generateOutputs(outputCount: Int, triggerFunction: TriggerFunction): List<Neuron> {
            val outputs = ArrayList<Neuron>()
            for (i in 0 until outputCount) {
                outputs.add(Neuron(
                        triggerFunction = triggerFunction,
                        name = "output$i"
                ))
            }
            return outputs
        }

        fun generateHiddenNeurons(outputCount: Int, order: Int, triggerFunction: TriggerFunction): List<Neuron> {
            val neurons = ArrayList<Neuron>()
            for (i in 0 until outputCount) {
                neurons.add(Neuron(
                        triggerFunction = triggerFunction,
                        name = "hidden $order-$i"
                ))
            }
            return neurons
        }

        // Extension
        fun NeuralNetwork.wireUp( inputs: Collection<ConnectionInput>, outputs: Collection<Neuron>) {
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

    fun build(outputCount: Int): NeuralNetwork {
        val network = generateNetwork(outputCount)

        val neurons = ArrayList<Collection<Neuron>>()
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
        return network
    }

    private fun generateNetwork(outputCount: Int): NeuralNetwork {
        val inputLayer = InputLayer(generateInputs(inputCount))
        val outputLayer = OutputLayer(generateOutputs(outputCount, triggerFunction))
        val network = NeuralNetwork(inputLayer, outputLayer)

        for (ind in 0 until hiddenLayerNeurons.size) {
            val hiddenLayer = generateHiddenNeurons(hiddenLayerNeurons[ind], ind, triggerFunction)
            network.addHiddenLayer(hiddenLayer)
        }

        return network
    }

}