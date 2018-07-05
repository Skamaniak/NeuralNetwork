package cz.maks.train

import cz.maks.model.NeuralNetwork
import cz.maks.util.ValidationUtils

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
class Trainer(
        private val network: NeuralNetwork,
        private val learningRate: Double = 0.3
) {

    private val hiddenNeurons = network.hiddenLayers
            .flatMap {it.neurons.toList()}
            .toTypedArray()

    private val reversedHiddenNeurons = hiddenNeurons
            .reversed()

    fun train(inputValues: DoubleArray, targets: DoubleArray, learningRate: Double) {
        network.evaluate(inputValues)
        adjust(targets, learningRate)
    }

    fun train(dataValue: DataValue) {
        ValidationUtils.validateCompatibility(dataValue, network)
        train(dataValue.inputs, dataValue.outputs, learningRate)
    }

    fun train(trainSet: TrainSet) {
        ValidationUtils.validateCompatibility(trainSet, network)
        for (dataValue in trainSet.data) {
            train(dataValue)
        }
    }

    fun train(trainSet: TrainSet, loops: Int) {
        ValidationUtils.validateLoopCount(loops)

        repeat(loops) { train(trainSet) }
    }

    fun train(trainSet: TrainSet, batchSize: Int, loops: Int = 1, epochs: Int = 1) {
        ValidationUtils.validateLoopCount(loops)
        ValidationUtils.validateBatchSize(trainSet, batchSize)

        repeat(epochs) {
            print("Epoch - $it ... ")
            var batch = trainSet.extractSubset(batchSize)
            repeat(loops) {
                train(batch)
                batch = trainSet.extractSubset(batchSize)
            }
            println("Mean Square Error: ${meanSquareError(batch)}")
        }
    }

    fun meanSquareError(dataValue: DataValue): Double {
        ValidationUtils.validateCompatibility(dataValue, network)

        network.evaluate(dataValue.inputs)

        var errorSum = 0.0
        for (ind in 0 until dataValue.outputCount) {
            errorSum += Math.pow(dataValue.outputs[ind] - network.outputs[ind].result, 2.0)
        }
        return errorSum / (2.0 * dataValue.outputCount)
    }

    fun meanSquareError(trainSet: TrainSet): Double {
        ValidationUtils.validateCompatibility(trainSet, network)

        var errorSum = 0.0
        for (dataValue in trainSet.data) {
            errorSum += meanSquareError(dataValue)
        }
        return errorSum / trainSet.data.size
    }

    private fun adjust(targets: DoubleArray, learningRate: Double) {
        computeErrorSignals(targets)
        adjust(learningRate)
    }

    private fun computeErrorSignals(targets: DoubleArray) {
        validateTargets(targets)

        for (ind in 0 until targets.size) {
            val outputNeuron = network.outputLayer.neurons[ind]
            outputNeuron.computeErrorSignal(targets[ind])
        }

        for (hiddenNeuron in reversedHiddenNeurons) {
            hiddenNeuron.computeErrorSignal()
        }
    }

    private fun adjust(learningRate: Double) {
        for (outputNeuron in network.outputLayer.neurons) {
            outputNeuron.adjustBias(learningRate)
        }

        for (hiddenNeuron in hiddenNeurons) {
            hiddenNeuron.adjustBias(learningRate)
        }

        for (connection in network.connections) {
            connection.adjustWeight(learningRate)
        }
    }


    private fun validateTargets(targets: DoubleArray) {
        val targetsCount = targets.size
        val outputNeuronsCount = network.outputLayer.neurons.size
        if (targetsCount != outputNeuronsCount) {
            throw IllegalArgumentException("Number of provided results $targetsCount doesn't match number of output " +
                    "neurons $outputNeuronsCount")
        }
    }
}