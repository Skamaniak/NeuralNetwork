package cz.maks

import cz.maks.data.MnistImageFile
import cz.maks.data.MnistLabelFile
import cz.maks.model.NeuralNetwork
import cz.maks.train.DataValue
import cz.maks.train.TrainSet
import java.io.EOFException

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
object NetworkDataSetUtils {
    fun createTrainSet(type: DataSetType): TrainSet {
        val trainSet = TrainSet(type.inputs, type.outputs)

        println("Loading $type")
        var imageSet: MnistImageFile? = null
        var labelSet: MnistLabelFile? = null
        try {
            imageSet = MnistImageFile(type.imageFilePath, "r")
            labelSet = MnistLabelFile(type.labelFilePath, "r")

            var ind = 0
            while (true) {
                ind++
                if (ind % 100 == 0) {
                    println("prepared: $ind")
                }

                val output = DoubleArray(type.outputs) {0.0}
                output[labelSet.readLabel() - type.startIndex] = 1.0

                val input = DoubleArray(type.inputs)
                for (j in 0 until type.inputs) {
                    input[j] = (imageSet.read().toDouble() / 256.0)
                }

                trainSet.addData(DataValue(input, output))
                imageSet.next()
                labelSet.next()
            }
        } catch (e: EOFException) {
            println("Loading finished")
        } finally {
            imageSet?.close()
            labelSet?.close()
        }

        return trainSet
    }

    fun testTrainSet(net: NeuralNetwork, set: TrainSet, printSteps: Int = 1000): Double {
        val setSize = set.data.size
        var correct = 0
        for (i in 0 until setSize) {
            val dataValue = set.data[i]
            val highest = indexOfHighestValue(net.evaluate(dataValue.inputs))


            val actualHighest = indexOfHighestValue(dataValue.outputs)
            if (highest == actualHighest) {
                correct++
            }
            if (i % printSteps == 0) {
                println(i.toString() + ": " + correct.toDouble() / (i + 1).toDouble())
            }
        }
        val successPerc = (correct.toDouble() / setSize) * 100
        println("Testing finished, RESULT: $correct / $setSize  -> $successPerc %")
        return successPerc
    }

    fun indexOfHighestValue(numbers: DoubleArray): Int {
        var highestValueIndex = 0
        var highestValue = Double.MIN_VALUE

        for (ind in 0 until numbers.size) {
            if (numbers[ind] > highestValue) {
                highestValue = numbers[ind]
                highestValueIndex = ind
            }
        }
        return highestValueIndex
    }
}
