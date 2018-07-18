package cz.maks

import cz.maks.model.Connection
import cz.maks.model.NeuralNetwork
import cz.maks.model.Neuron
import cz.maks.strategies.BiasInitialisationFunction
import cz.maks.strategies.LossFunction
import cz.maks.strategies.WeightInitialisationFunction
import cz.maks.util.ValidationUtils

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */

fun NeuralNetwork.evaluate() {
    for (hiddenLayer in hiddenLayers) {
        for (hiddenNeuron in hiddenLayer.neurons) {
            hiddenNeuron.computeOutput()
        }
    }

    for (ind in 0 until outputCount) {
        outputs[ind].computeOutput()
    }
}

fun NeuralNetwork.setInputs(inputValues: DoubleArray) {
    ValidationUtils.validateInputValues(inputValues, this)
    for (ind in 0 until inputValues.size) {
        inputLayer.inputs[ind].inputValue = inputValues[ind]
    }
}

fun NeuralNetwork.extractResults(): DoubleArray {
    val results = DoubleArray(outputCount)
    for (ind in 0 until outputCount) {
        results[ind] = outputs[ind].result
    }

    return results
}

fun NeuralNetwork.init() {
    for (hiddenLayer in hiddenLayers) {
        for (hiddenNeuron in hiddenLayer.neurons) {
            hiddenNeuron.initBias(biasInitialisationFunction)
            hiddenNeuron.initWeights(weightInitialisationFunction)
        }
    }

    for (ind in 0 until outputCount) {
        outputs[ind].initBias(biasInitialisationFunction)
        outputs[ind].initWeights(weightInitialisationFunction)
    }
}


fun Connection.computeWeightedResult(): Double {
    return input.getValue() * weight
}

fun Connection.computeWeightedErrorSignal(): Double {
    return output.errorSignal * weight
}

fun Connection.adjustWeight(learningRate: Double) {
    val delta = -learningRate * input.getValue() * output.errorSignal
    weight += delta
}


fun Neuron.computeOutput() {
    var summedInputs = 0.0
    for (con in inputConnections) {
        summedInputs += con.computeWeightedResult()
    }
    result = activationFunction.apply(summedInputs + bias)
    outputDerivative = activationFunction.derivative(result)
}

fun Neuron.adjustBias(learningRate: Double) {
    val delta = -learningRate * errorSignal
    bias += delta
}

fun Neuron.initWeights(weightInitialisationFunction: WeightInitialisationFunction) {
    val inputCount = inputConnections.size
    val outputCount = outputConnections.size
    for (inConnection in inputConnections) {
        inConnection.weight = weightInitialisationFunction.apply(inputCount, outputCount)
    }
}

fun Neuron.initBias(biasInitialisationFunction: BiasInitialisationFunction) {
    bias = biasInitialisationFunction.apply()
}

/**
 * Computes error signal if the target value is not available. This is typically used for the hidden layer neurons
 */
fun Neuron.computeErrorSignal() {
    var sumOfWeightedErrors = 0.0
    for (con in outputConnections) {
        sumOfWeightedErrors += con.computeWeightedErrorSignal()
    }
    errorSignal = sumOfWeightedErrors * outputDerivative
}

/**
 * Computes error signal if the target value is available. This is typically used for the output layer neurons
 */
fun Neuron.computeErrorSignal(target: Double, lossFunction: LossFunction) {
    errorSignal = lossFunction.apply(result, target) * outputDerivative
}
