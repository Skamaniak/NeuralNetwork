import cz.maks.data.MnistImageFile
import cz.maks.data.MnistLabelFile
import cz.maks.model.DenseNetworkBuilder
import cz.maks.model.NeuralNetwork
import cz.maks.model.TriggerFunction
import cz.maks.persistence.FilePersistence
import cz.maks.train.DataValue
import cz.maks.train.TrainSet
import cz.maks.train.Trainer

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
fun main(args: Array<String>) {
    val network = DenseNetworkBuilder(28 * 28, TriggerFunction.SIGMOID)
            .addHiddenLayer(70)
            .addHiddenLayer(35)
            .build(10)

    val trainSet = createTrainSet(0, 29999, DataSetType.TRAIN)

    val trainer = Trainer(network, 0.3)
    trainer.train(trainSet, 50, 50, 100)

    val testSet = createTrainSet(0, 9999, DataSetType.TRAIN)
    testTrainSet(network, testSet, 10)
}

enum class DataSetType(val imageFilePath: String, val labelFilePath: String) {
    TRAIN(
            "Mnist/src/main/resources/train/trainImage.idx3-ubyte",
            "Mnist/src/main/resources/train/trainLabel.idx1-ubyte"
    ),
    TEST(
            "Mnist/src/main/resources/test/t10k-mages.idx3-ubyte",
            "Mnist/src/main/resources/test/t10k-labels.idx1-ubyte"
    )
}


fun createTrainSet(start: Int, end: Int, type: DataSetType): TrainSet {
    val trainSet = TrainSet(28 * 28, 10)

    try {
        val imageSet = MnistImageFile(type.imageFilePath, "r")
        val labelSet = MnistLabelFile(type.labelFilePath, "r")

        for (i in start..end) {
            if (i % 100 == 0) {
                println("prepared: $i")
            }

            val output = mutableListOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            output[labelSet.readLabel()] = 1.0

            val input = ArrayList<Double>(28 * 28)
            for (j in 0 until 28 * 28) {
                input.add(j, (imageSet.read().toDouble() / 256.0))
            }

            trainSet.addData(DataValue(input, output))
            imageSet.next()
            labelSet.next()
        }
    } catch (e: Exception) {
        e.printStackTrace()
    }

    return trainSet
}

fun testTrainSet(net: NeuralNetwork, set: TrainSet, printSteps: Int) {
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
    println("Testing finished, RESULT: $correct / $setSize  -> ${(correct.toDouble() / setSize) * 100} %")
}

fun indexOfHighestValue(numbers: List<Double>): Int {
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