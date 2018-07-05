package cz.maks.model

import cz.maks.util.ValidationUtils
import java.util.*

interface ConnectionInput {
    fun getValue(): Double
    fun getIdentifier(): String
}

object RandomUtils {
    private val RANDOM = Random()

    fun generateRandom(lower: Double, upper: Double): Double {
        return RANDOM.nextDouble() * (upper - lower) + lower
    }
}

class Neuron(
        val id: String = UUID.randomUUID().toString(),
        val name: String = "unknown",
        var result: Double = 0.0,
        var errorSignal: Double = 0.0,
        var outputDerivative: Double = 0.0,
        var bias: Double = RandomUtils.generateRandom(-0.5, 0.7),
        val triggerFunction: TriggerFunction = TriggerFunction.SIGMOID,
        private var inputConnections: Array<Connection> = emptyArray(),
        private var outputConnections: Array<Connection> = emptyArray()
) : ConnectionInput {

    override fun getValue(): Double = result
    override fun getIdentifier(): String = id

    fun addInputConnection(connection: Connection) {
        inputConnections = inputConnections.plus(connection)
    }

    fun addOutputConnection(connection: Connection) {
        outputConnections = outputConnections.plus(connection)
    }

    fun computeOutput() {
        var summedInputs = 0.0
        for (con in inputConnections) {
            summedInputs += con.computeWeightedResult()
        }
        result = triggerFunction.apply(summedInputs + bias)
        outputDerivative = computeOutputDerivative()
    }

    fun adjustBias(learningRate: Double) {
        val delta = -learningRate * errorSignal
        bias += delta
    }

    /**
     * Computes error signal if the target value is not available. This is typically used for the hidden layer neurons
     */
    fun computeErrorSignal() {
        var sumOfWeightedErrors = 0.0
        for (con in outputConnections) {
            sumOfWeightedErrors += con.computeWeightedErrorSignal()
        }
        errorSignal = sumOfWeightedErrors * outputDerivative
    }

    /**
     * Computes error signal if the target value is available. This is typically used for the output layer neurons
     */
    fun computeErrorSignal(target: Double) {
        errorSignal = (result - target) * outputDerivative
    }

    private fun computeOutputDerivative(): Double {
        return result * (1 - result)
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
        return "Connection $input => $output, weight=$weight)"
    }
}

class InputLayer(val inputs: Array<Input>)

class HiddenLayer(val neurons: Array<Neuron>)

class OutputLayer(val neurons: Array<Neuron>)

data class NeuralNetwork(
        var inputLayer: InputLayer,
        var outputLayer: OutputLayer,
        var hiddenLayers: Array<HiddenLayer> = emptyArray(),
        var connections: Array<Connection> = emptyArray()
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

    fun setInputs(inputValues: DoubleArray) {
        ValidationUtils.validateInputValues(inputValues, this)
        for (ind in 0 until inputValues.size) {
            inputLayer.inputs[ind].inputValue = inputValues[ind]
        }
    }

    fun extractResults(): DoubleArray {
        val results = DoubleArray(outputCount)
        for (ind in 0 until outputCount) {
            results[ind] = outputs[ind].result
        }

        return results
    }

    fun evaluate() {
        for (hiddenLayer in hiddenLayers) {
            for (hiddenNeuron in hiddenLayer.neurons) {
                hiddenNeuron.computeOutput()
            }
        }

        for (ind in 0 until outputCount) {
            outputs[ind].computeOutput()
        }
    }

    fun evaluate(inputValues: DoubleArray): DoubleArray {
        setInputs(inputValues)
        evaluate()
        return extractResults()
    }


}
