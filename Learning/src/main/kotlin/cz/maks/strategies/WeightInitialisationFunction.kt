package cz.maks.strategies

import java.util.*

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
enum class WeightInitialisationFunction {
    XAVIER_NORMAL {
        override fun apply(inputs: Int, outputs: Int, rand: Random): Double {
            val standardDeviation = Math.sqrt(2 / (inputs.toDouble() + outputs.toDouble()))
            return rand.nextGaussian() * standardDeviation
        }
    },
    HE_NORMAL {
        override fun apply(inputs: Int, outputs: Int, rand: Random): Double {
            val standardDeviation = Math.sqrt(2 / inputs.toDouble())
            return rand.nextGaussian() * standardDeviation
        }
    },
    LECUN_NORMAL {
        override fun apply(inputs: Int, outputs: Int, rand: Random): Double {
            val standardDeviation = Math.sqrt(1 / inputs.toDouble())
            return rand.nextGaussian() * standardDeviation
        }
    },
    RANDOM {
        override fun apply(inputs: Int, outputs: Int, rand: Random): Double {
            return (rand.nextDouble() * 2) - 1
        }

    };

    abstract fun apply(inputs: Int, outputs: Int, rand: Random = Random()): Double
}