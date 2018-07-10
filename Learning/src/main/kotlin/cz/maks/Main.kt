package cz.maks

import cz.maks.builder.DenseNetworkBuilder
import cz.maks.strategies.ActivationFunction
import cz.maks.persistence.FilePersistence
import cz.maks.train.DataValue
import cz.maks.train.TrainSet
import cz.maks.train.Trainer
import java.util.*

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
fun main(args: Array<String>) {
    var neuralNetwork = DenseNetworkBuilder(2, ActivationFunction.TANH)
            .addHiddenLayer(3)
            .build(1)

    val trainSet = TrainSet(2, 1)
            .addData(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0))
            .addData(doubleArrayOf(1.0, 0.0), doubleArrayOf(1.0))
            .addData(doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0))
            .addData(doubleArrayOf(1.0, 1.0), doubleArrayOf(0.0))

    val trainer = Trainer(neuralNetwork)
    println(trainer.meanSquareError(DataValue(doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0))))
    trainer.train(trainSet, 100000)
    println(trainer.meanSquareError(DataValue(doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0))))

    FilePersistence.store(neuralNetwork, "test.zip")
    neuralNetwork = FilePersistence.load("test.zip")

    printArray(neuralNetwork.evaluate(doubleArrayOf(0.0, 0.0)))
    printArray(neuralNetwork.evaluate(doubleArrayOf(0.0, 1.0)))
    printArray(neuralNetwork.evaluate(doubleArrayOf(1.0, 0.0)))
    printArray(neuralNetwork.evaluate(doubleArrayOf(1.0, 1.0)))

    for (conn in neuralNetwork.connections) {
        println(conn)
    }
}

private fun printArray(a: DoubleArray) {
    println(Arrays.toString(a))
}