package cz.maks.model

import kotlin.math.E
import kotlin.math.pow


/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
enum class ActivationFunction {
    SIGMOID {
        override fun apply(sum: Double): Double {
            return 1.0 / (1.0 + E.pow(-sum))
        }
    },
    TANH {
        override fun apply(sum: Double): Double {
            return (2.0 / (1.0 + E.pow(-2 * sum))) - 1
        }
    },
    RELU {
        override fun apply(sum: Double): Double {
            return if (sum >= 0) sum else 0.0
        }
    },
    LEAKY_RELU {
        override fun apply(sum: Double): Double {
            return if (sum >= 0) sum else sum * 0.01
        }
    };

    abstract fun apply(sum: Double): Double
}