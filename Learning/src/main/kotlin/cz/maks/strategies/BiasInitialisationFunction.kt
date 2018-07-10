package cz.maks.strategies

import java.util.*

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
enum class BiasInitialisationFunction {
    ZERO {
        override fun apply(rand: Random): Double = 0.0
    },
    RANDOM {
        override fun apply(rand: Random): Double {
            return (rand.nextDouble() * 1.2) - 0.5 //TODO
        }

    };

    abstract fun apply(rand: Random): Double

}