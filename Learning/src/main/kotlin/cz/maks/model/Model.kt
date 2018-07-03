package cz.maks.model

import cz.maks.util.ValidationUtils
import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashSet

interface ConnectionInput {
    fun getValue(): Double
}

object RandomUtils {
    private val RANDOM = Random()

    fun generateRandom(lower: Double, upper: Double): Double {
        return RANDOM.nextDouble() * (upper - lower) + lower
    }
}

class Neuron(
        private val name: String = "unknown",
        private val triggerFunction: TriggerFunction,
        private var inputConnections: MutableCollection<Connection> = HashSet(),
        private var outputConnections: MutableCollection<Connection> = HashSet(),
        var result: Double = 0.0,
        var errorSignal: Double = 0.0,
        var bias: Double = RandomUtils.generateRandom(-0.5, 0.7)
) : ConnectionInput {

    override fun getValue(): Double {
        return result
    }

    fun addInputConnection(connection: Connection) {
        inputConnections.add(connection)
    }

    fun addOutputConnection(connection: Connection) {
        outputConnections.add(connection)
    }

    fun computeOutput() {
        val summedInputs = inputConnections
                .map(Connection::computeWeightedResult)
                .sum()
        result = triggerFunction.apply(summedInputs + bias)
    }

    fun adjustBias(learningRate: Double) {
        val delta = -learningRate * errorSignal
        bias += delta
    }

    /**
     * Computes error signal if the target value is not available. This is typically used for the hidden layer neurons
     */
    fun computeErrorSignal() {
        val sumOfWeightedErrors = outputConnections
                .map(Connection::computeWeightedErrorSignal)
                .sum()
        errorSignal = sumOfWeightedErrors * computeOutputDerivative()
    }

    /**
     * Computes error signal if the target value is available. This is typically used for the output layer neurons
     */
    fun computeErrorSignal(target: Double) {
        errorSignal = (result - target) * computeOutputDerivative()
    }

    private fun computeOutputDerivative(): Double {
        return result * (1 - result)
    }

    override fun toString(): String {
        return "$name - bias: $bias"
    }
}

class Input(inputValue: Double = 0.0) : ConnectionInput {
    var inputValue: Double = inputValue
        set(newVal) {
            ValidationUtils.validateInputValue(newVal)
            field = newVal
        }

    override fun getValue(): Double {
        return inputValue
    }
}

class Connection(
        private val input: ConnectionInput,
        private val output: Neuron,
        private var weight: Double = RandomUtils.generateRandom(-1.0, 1.0)
) {

    fun computeWeightedResult(): Double {
        return input.getValue() * weight
    }

    fun computeWeightedErrorSignal(): Double {
        return output.errorSignal * weight
    }

    fun adjustWeight(learningRate: Double) {
        val delta = -learningRate * input.getValue() * output.errorSignal
        weight += delta
    }

    override fun toString(): String {
        return "Connection(input=$input, output=$output, weight=$weight)"
    }
}


class InputLayer(val inputs: List<Input>)

class HiddenLayer(val neurons: Collection<Neuron>)

class OutputLayer(val neurons: List<Neuron>)

data class NeuralNetwork(
        var inputLayer: InputLayer,
        var outputLayer: OutputLayer,
        var hiddenLayers: MutableList<HiddenLayer> = ArrayList(),
        var connections: MutableCollection<Connection> = HashSet()
) {
    val inputs = inputLayer.inputs
    val outputs = outputLayer.neurons
    val inputCount = inputs.size
    val outputCount = outputs.size

    fun addHiddenLayer(neurons: Collection<Neuron>) {
        hiddenLayers.add(HiddenLayer(neurons))
    }

    fun connect(input: ConnectionInput, output: Neuron) {
        val connection = Connection(input, output)

        if (input is Neuron) {
            input.addOutputConnection(connection)
        }
        output.addInputConnection(connection)
        connections.add(connection)
    }


    fun setInputs(inputValues: List<Double>) {
        ValidationUtils.validateInputValues(inputValues, this)
        for (ind in 0 until inputValues.size) {
            inputLayer.inputs[ind].inputValue = inputValues[ind]
        }
    }

    fun extractOutputs(): List<Double> {
        return outputLayer.neurons
                .map(Neuron::result)
    }

    fun evaluate() {
        hiddenLayers
                .flatMap(HiddenLayer::neurons)
                .forEach(Neuron::computeOutput)

        outputLayer.neurons
                .forEach(Neuron::computeOutput)
    }

    fun evaluate(inputValues: List<Double>): List<Double> {
        setInputs(inputValues)
        evaluate()
        return extractOutputs()
    }


}
