package cz.maks.model

import cz.maks.computeOutput
import cz.maks.evaluate
import cz.maks.extractResults
import cz.maks.setInputs
import cz.maks.strategies.*
import cz.maks.util.ValidationUtils
import org.apache.logging.log4j.core.util.NetUtils
import java.util.*

interface ConnectionInput {
    fun getValue(): Double
    fun getIdentifier(): String
}


class Neuron(
        val id: String = UUID.randomUUID().toString(),
        val name: String = "unknown",
        var result: Double = 0.0,
        var errorSignal: Double = 0.0,
        var outputDerivative: Double = 0.0,
        var bias: Double = RandomUtils.generateRandom(-0.5, 0.7),
        val activationFunction: ActivationFunction = Activation.sigmoid(),
        var inputConnections: Array<Connection> = emptyArray(),
        var outputConnections: Array<Connection> = emptyArray()
) : ConnectionInput {

    override fun getValue(): Double = result
    override fun getIdentifier(): String = id

    fun addInputConnection(connection: Connection) {
        inputConnections = inputConnections.plus(connection)
    }

    fun addOutputConnection(connection: Connection) {
        outputConnections = outputConnections.plus(connection)
    }

    override fun toString(): String {
        return "Neuron $name, bias: $bias"
    }
}

class Input(
        val id: String = UUID.randomUUID().toString(),
        inputValue: Double = 0.0
) : ConnectionInput {
    var inputValue: Double = inputValue
        set(newVal) {
            ValidationUtils.validateInputValue(newVal)
            field = newVal
        }

    override fun getValue(): Double = inputValue
    override fun getIdentifier(): String = id
}

class Connection(
        val input: ConnectionInput,
        val output: Neuron,
        var weight: Double = RandomUtils.generateRandom(-1.0, 1.0)
) {

    init {
        output.addInputConnection(this)
        if (input is Neuron) {
            input.addOutputConnection(this)
        }
    }

    override fun toString(): String {
        return "Connection $input => $output, weight=$weight)"
    }
}

class InputLayer(val inputs: Array<Input>)

class HiddenLayer(val neurons: Array<Neuron>)

class OutputLayer(val neurons: Array<Neuron>)

class NeuralNetwork(
        var inputLayer: InputLayer,
        var outputLayer: OutputLayer,
        var hiddenLayers: Array<HiddenLayer> = emptyArray(),
        var connections: Array<Connection> = emptyArray(),
        var weightInitialisationFunction: WeightInitialisationFunction = WeightInitialisation.xavierNormal(),
        var biasInitialisationFunction: BiasInitialisationFunction = BiasInitialisation.zeros(),
        var lossFunction: LossFunction = Loss.difference()
) {
    val inputs = inputLayer.inputs
    val outputs = outputLayer.neurons
    val inputCount = inputs.size
    val outputCount = outputs.size

    fun addHiddenLayer(neurons: Array<Neuron>) {
        hiddenLayers = hiddenLayers.plus(HiddenLayer(neurons))
    }

    fun connect(con: Connection) {
        connections = connections.plus(con)
    }

    fun connect(input: ConnectionInput, output: Neuron) {
        connect(Connection(input, output))
    }

    fun evaluate(inputValues: DoubleArray): DoubleArray {
        setInputs(inputValues)
        evaluate()
        return extractResults()
    }
}
