package cz.maks.util

import cz.maks.model.NeuralNetwork
import cz.maks.train.DataValue
import cz.maks.train.TrainSet

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */

object ValidationUtils {
    fun validateCompatibility(trainSet: TrainSet, network: NeuralNetwork) {
        if (trainSet.inputCount != network.inputCount) {
            throw IllegalArgumentException("Provided test set input count ${trainSet.inputCount} doesn't match " +
                    "neural network input count ${network.inputCount}")
        }
        if (trainSet.outputCount != network.outputCount) {
            throw IllegalArgumentException("Provided test set output count ${trainSet.outputCount} doesn't match " +
                    "neural network output count ${network.outputCount}")
        }
    }

    fun validateCompatibility(dataValue: DataValue, network: NeuralNetwork) {
        if (dataValue.inputCount != network.inputCount) {
            throw IllegalArgumentException("Provided data value input count ${dataValue.inputCount} doesn't match " +
                    "neural network input count ${network.inputCount}")
        }
        if (dataValue.outputCount != network.outputCount) {
            throw IllegalArgumentException("Provided data value output count ${dataValue.outputCount} doesn't match " +
                    "neural network output count ${network.outputCount}")
        }
    }

    fun validateInputValues(inputValues: Collection<Double>, network: NeuralNetwork) {
        if (inputValues.size != network.inputCount) {
            throw IllegalArgumentException("Number of provided input values ${inputValues.size} doesn't match " +
                    "number of neural network inputs ${network.inputCount}")
        }

        val allValuesValid = inputValues.all(this::isValid)
        if (!allValuesValid) {
            throw IllegalArgumentException("Found input value which is out of interval [0, 1]")
        }
    }

    fun validateBatchSize(trainSet: TrainSet, batchSize: Int) {
        if (batchSize <= 0) {
            throw IllegalArgumentException("Batch size needs to be greater than 0 but was $batchSize")
        }

        val trainDataSize = trainSet.data.size
        if (trainDataSize < batchSize) {
            throw IllegalArgumentException("Batch size $batchSize is greater than train set size $trainDataSize")
        }
    }

    fun validateLoopCount(loopCount: Int) {
        if (loopCount <= 0) {
            throw IllegalArgumentException("Loop count needs to be greater than 0 but was $loopCount")
        }
    }

    fun validateInputValue(value: Double) {
        if (!isValid(value)) {
            throw IllegalArgumentException("Invalid input value $value")
        }
    }

    private fun isValid(input: Double): Boolean {
        return input in 0.0..1.0
    }
}