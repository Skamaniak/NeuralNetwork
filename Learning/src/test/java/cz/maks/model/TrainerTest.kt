package cz.maks.model

import cz.maks.builder.DenseNetworkBuilder
import cz.maks.evaluate
import cz.maks.strategies.Activation
import cz.maks.strategies.ActivationFunction
import cz.maks.train.TrainSet
import cz.maks.train.Trainer
import org.assertj.core.api.Assertions
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
@DisplayName("Tests neural network trainer")
class TrainerTest {

    @DisplayName("Train XOR")
    @Test
    fun testTrainDenseNeuralNetworkToXor() {
        val neuralNetwork = DenseNetworkBuilder(2, Activation.sigmoid())
                .addHiddenLayer(3)
                .addHiddenLayer(3)
                .addHiddenLayer(3)
                .build(1)

        val trainSet = TrainSet(2, 1)
                .addData(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0))
                .addData(doubleArrayOf(1.0, 0.0), doubleArrayOf(1.0))
                .addData(doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0))
                .addData(doubleArrayOf(1.0, 1.0), doubleArrayOf(0.0))

        val trainer = Trainer(neuralNetwork)
        trainer.train(trainSet, 2, 1000, 100)

        Assertions.assertThat(neuralNetwork.evaluate(doubleArrayOf(0.0, 0.0))[0])
                .isBetween(0.0, 0.05)
        Assertions.assertThat(neuralNetwork.evaluate(doubleArrayOf(0.0, 1.0))[0])
                .isBetween(0.95, 1.0)
        Assertions.assertThat(neuralNetwork.evaluate(doubleArrayOf(1.0, 0.0))[0])
                .isBetween(0.95, 1.0)
        Assertions.assertThat(neuralNetwork.evaluate(doubleArrayOf(1.0, 1.0))[0])
                .isBetween(0.0, 0.05)

    }

}