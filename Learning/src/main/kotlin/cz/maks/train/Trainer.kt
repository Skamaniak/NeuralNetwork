package cz.maks.train

import cz.maks.model.HiddenLayer
import cz.maks.model.NeuralNetwork
import cz.maks.model.Neuron
import cz.maks.util.ValidationUtils

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
class Trainer(
        private val network: NeuralNetwork,
        private val learningRate: Double = 0.3
) {

    fun train(inputValues: List<Double>, targets: List<Double>, learningRate: Double) {
        network.evaluate(inputValues)
        adjust(targets, learningRate)
    }

    fun train(dataValue: DataValue) {
        ValidationUtils.validateCompatibility(dataValue, network)
        train(dataValue.inputs, dataValue.outputs, learningRate)
    }

    fun train(trainSet: TrainSet) {
        ValidationUtils.validateCompatibility(trainSet, network)
        trainSet.data
                .forEach(this::train)
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

        val acc = (0 until dataValue.outputCount)
                .map { Math.pow(dataValue.outputs[it] - network.outputs[it].result, 2.0) }
                .sum()
        return acc / (2.0 * dataValue.outputCount)
    }

    fun meanSquareError(trainSet: TrainSet): Double {
        ValidationUtils.validateCompatibility(trainSet, network)

        val acc = trainSet.data
                .map { meanSquareError(it) }
                .sum()
        return acc / trainSet.data.size
    }

    private fun adjust(targets: List<Double>, learningRate: Double) {
        computeErrorSignals(targets)
        adjust(learningRate)
    }

    private fun computeErrorSignals(targets: List<Double>) {
        validateTargets(targets)

        for (ind in 0 until targets.size) {
            val outputNeuron = network.outputLayer.neurons[ind]
            outputNeuron.computeErrorSignal(targets[ind])
        }

        network.hiddenLayers
                .reversed()
                .flatMap(HiddenLayer::neurons)
                .forEach(Neuron::computeErrorSignal)
    }

    private fun adjust(learningRate: Double) {
        network.outputLayer.neurons
                .forEach({ it.adjustBias(learningRate) })

        network.hiddenLayers
                .flatMap(HiddenLayer::neurons)
                .forEach({ it.adjustBias(learningRate) })

        network.connections
                .forEach({ it.adjustWeight(learningRate) })
    }


    private fun validateTargets(targets: Collection<Double>) {
        val targetsCount = targets.size
        val outputNeuronsCount = network.outputLayer.neurons.size
        if (targetsCount != outputNeuronsCount) {
            throw IllegalArgumentException("Number of provided results $targetsCount doesn't match number of output " +
                    "neurons $outputNeuronsCount")
        }
    }
}