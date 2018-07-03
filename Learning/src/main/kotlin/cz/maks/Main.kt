package cz.maks

import cz.maks.model.DenseNetworkBuilder
import cz.maks.model.TriggerFunction
import cz.maks.train.DataValue
import cz.maks.train.TrainSet
import cz.maks.train.Trainer

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
fun main(args: Array<String>) {
    val neuralNetwork = DenseNetworkBuilder(2,TriggerFunction.SIGMOID)
            .addHiddenLayer(3)
            .build(1)

    val trainSet = TrainSet(2, 1)
            .addData(listOf(0.0, 0.0), listOf(0.0))
            .addData(listOf(1.0, 0.0), listOf(1.0))
            .addData(listOf(0.0, 1.0), listOf(1.0))
            .addData(listOf(1.0, 1.0), listOf(0.0))

    val trainer = Trainer(neuralNetwork)
    println(trainer.meanSquareError(DataValue(listOf(0.0, 1.0), listOf(1.0))))
    trainer.train(trainSet, 100000)
    println(trainer.meanSquareError(DataValue(listOf(0.0, 1.0), listOf(1.0))))

    println(neuralNetwork.evaluate(listOf(0.0, 0.0)))
    println(neuralNetwork.evaluate(listOf(0.0, 1.0)))
    println(neuralNetwork.evaluate(listOf(1.0, 0.0)))
    println(neuralNetwork.evaluate(listOf(1.0, 1.0)))

    for (conn in neuralNetwork.connections) {
        println(conn)
    }
}