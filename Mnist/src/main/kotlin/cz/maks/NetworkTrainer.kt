package cz.maks

import cz.maks.builder.DenseNetworkBuilder
import cz.maks.strategies.ActivationFunction
import cz.maks.model.NeuralNetwork
import cz.maks.persistence.FilePersistence
import cz.maks.train.Trainer
import java.util.concurrent.Executors
import kotlin.system.measureTimeMillis

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
fun main(args: Array<String>) {
//    TrainNetwork().trainParallel(DataSetType.TRAIN_DIGITS, DataSetType.TEST_DIGITS, 180, 180)
    TrainNetwork().trainParallel(DataSetType.TRAIN_LETTERS, DataSetType.TEST_LETTERS, 179, 180)
}

class TrainNetwork {
    fun trainParallel(trainDataSet: DataSetType, testDataSet: DataSetType, leastNeurons: Int, mostNeurons: Int) {
        val trainSet = NetworkDataSetUtils.createTrainSet(trainDataSet)
        val testSet = NetworkDataSetUtils.createTrainSet(testDataSet)

        val executorService = Executors.newFixedThreadPool(4)

        for (neurons in leastNeurons until mostNeurons) {
            println("Scheduling train task for $neurons neurons in hidden layer")
            executorService.submit {
                println("${Thread.currentThread().name} Starting training for $neurons")
                val network = DenseNetworkBuilder(trainDataSet.inputs, ActivationFunction.SIGMOID)
                        .addHiddenLayer(neurons)
                        .addHiddenLayer(neurons / 2)
                        .build(trainDataSet.outputs)

                val took = measureTimeMillis {
                    val trainer = Trainer(network, 0.3)
                    var highest = 85.0
                    trainer.train(
                            trainSet = trainSet,
                            batchSize = 100,
                            loops = 200,
                            epochs = 100,
                            epochListener = {
                                val successRatePerc = NetworkDataSetUtils.testTrainSet(network, testSet)
                                if (successRatePerc > highest) {
                                    highest = successRatePerc
                                    FilePersistence.store(network,
                                            "${trainDataSet.dataSetName}#${network.structure()}#e$it-s${formatToTwoDecimals(successRatePerc)}.zip")
                                } else {
                                    println("${Thread.currentThread().name} Storing skipped, success rate $successRatePerc is too low.")
                                }
                            })
                }
                println("${Thread.currentThread().name} Took: $took ms")
            }
        }
    }

    private fun formatToTwoDecimals(number: Double): String {
        return String.format("%.2f", number)
    }

    private fun NeuralNetwork.structure(): String {
        val nodes = ArrayList<Int>()

        nodes.add(inputCount)
        hiddenLayers.forEach {
            nodes.add(it.neurons.size)
        }
        nodes.add(outputCount)

        return nodes.joinToString(separator = "-")
    }
}