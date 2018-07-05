package cz.maks.model

import cz.maks.ex.MalformedNetworkException
import java.util.*

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */


data class NeuronDefinition(
        val id: String = UUID.randomUUID().toString(),
        val name: String = "unknown",
        val triggerFunction: TriggerFunction = TriggerFunction.SIGMOID,
        val bias: Double = 0.0
)

data class InputDefinition(val id: String = UUID.randomUUID().toString())

data class ConnectionDefinition(
        val input: String = "unknown",
        val output: String = "unknown",
        var weight: Double = 0.0
)

data class InputLayerDefinition(val inputs: List<InputDefinition> = emptyList())
data class HiddenLayerDefinition(val neurons: List<NeuronDefinition> = emptyList())
data class OutputLayerDefinition(val neurons: List<NeuronDefinition> = emptyList())

data class NeuralNetworkDefinition(
        val inputLayer: InputLayerDefinition = InputLayerDefinition(),
        val hiddenLayers: List<HiddenLayerDefinition> = emptyList(),
        val outputLayer: OutputLayerDefinition = OutputLayerDefinition(),
        val connections: Collection<ConnectionDefinition> = emptySet()
)

object SerialisationModel {
    fun fromComputationalModel(network: NeuralNetwork): NeuralNetworkDefinition {
        val inputLayerDefinition = fromComputationalModel(network.inputLayer)
        val hiddenLayersDefinition = network.hiddenLayers.map { fromComputationalModel(it) }
        val outputLayerDefinition = fromComputationalModel(network.outputLayer)
        val connections = network.connections.map { fromComputationalModel(it) }

        return NeuralNetworkDefinition(
                inputLayer = inputLayerDefinition,
                hiddenLayers = hiddenLayersDefinition,
                outputLayer = outputLayerDefinition,
                connections = connections
        )
    }

    private fun fromComputationalModel(input: Input): InputDefinition {
        return InputDefinition(input.id)
    }

    private fun fromComputationalModel(neuron: Neuron): NeuronDefinition {
        return NeuronDefinition(
                id = neuron.id,
                name = neuron.name,
                triggerFunction = neuron.triggerFunction,
                bias = neuron.bias)
    }

    private fun fromComputationalModel(connection: Connection): ConnectionDefinition {
        return ConnectionDefinition(
                input = connection.input.getIdentifier(),
                output = connection.output.getIdentifier(),
                weight = connection.weight
        )
    }

    private fun fromComputationalModel(inputLayer: InputLayer): InputLayerDefinition {
        val inputDefinitions = inputLayer.inputs
                .map { fromComputationalModel(it) }
        return InputLayerDefinition(inputDefinitions)
    }

    private fun fromComputationalModel(hiddenLayer: HiddenLayer): HiddenLayerDefinition {
        val neuronDefinitions = hiddenLayer.neurons
                .map { fromComputationalModel(it) }
        return HiddenLayerDefinition(neuronDefinitions)
    }

    private fun fromComputationalModel(outputLayer: OutputLayer): OutputLayerDefinition {
        val neuronDefinitions = outputLayer.neurons
                .map { fromComputationalModel(it) }
        return OutputLayerDefinition(neuronDefinitions)
    }

    fun toComputationalModel(network: NeuralNetworkDefinition): NeuralNetwork {
        val inputLayer = toComputationalModel(network.inputLayer)
        val outputLayer = toComputationalModel(network.outputLayer)
        val hiddenLayers = network.hiddenLayers.map { toComputationalModel(it) }

        val result = NeuralNetwork(
                inputLayer = inputLayer,
                hiddenLayers = hiddenLayers.toTypedArray(),
                outputLayer = outputLayer
        )

        val idToInput = inputLayer.inputs
                .map { it.id to it }
                .toMap()

        val idToNeuron = hiddenLayers
                .flatMap { it.neurons.toList() }
                .plus(outputLayer.neurons)
                .map { it.id to it }
                .toMap()

        fun getConnectionInput(id: String): ConnectionInput {
            var input: ConnectionInput? = idToInput[id]
            input = input ?: idToNeuron[id]
            return input ?: throw MalformedNetworkException("Connection input with id $id not found")
        }

        fun getConnectionOutput(id: String): Neuron {
            return idToNeuron[id] ?: throw MalformedNetworkException("Connection output with id $id not found")
        }

        network.connections
                .map {
                    Connection(
                            input = getConnectionInput(it.input),
                            output = getConnectionOutput(it.output),
                            weight = it.weight
                    )
                }
                .forEach(result::connect)

        return result
    }

    private fun toComputationalModel(input: InputDefinition): Input {
        return Input(input.id)
    }


    private fun toComputationalModel(neuron: NeuronDefinition): Neuron {
        return Neuron(
                id = neuron.id,
                name = neuron.name,
                bias = neuron.bias,
                triggerFunction = neuron.triggerFunction
        )
    }

    private fun toComputationalModel(hiddenLayer: HiddenLayerDefinition): HiddenLayer {
        val neurons = hiddenLayer.neurons
                .map { toComputationalModel(it) }
        return HiddenLayer(neurons.toTypedArray())
    }

    private fun toComputationalModel(outputLayer: OutputLayerDefinition): OutputLayer {
        val outputs = outputLayer.neurons
                .map { toComputationalModel(it) }
        return OutputLayer(outputs.toTypedArray())
    }

    private fun toComputationalModel(inputLayer: InputLayerDefinition): InputLayer {
        val inputs = inputLayer.inputs
                .map { toComputationalModel(it) }

        return InputLayer(inputs.toTypedArray())
    }
}