package cz.maks.strategies

import kotlin.math.E
import kotlin.math.pow


/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
enum class ActivationFunction {
    SIGMOID {
        override fun derivative(value: Double): Double = value * (1 - value)
        override fun apply(value: Double): Double = 1.0 / (1.0 + E.pow(-value))
    },
    TANH {
        override fun derivative(value: Double): Double = 1 - Math.pow(Math.tanh(value), 2.0)
        override fun apply(value: Double): Double = Math.tanh(value)
    },
    RECTIFIED_LINEAR_UNIT {
        override fun derivative(value: Double): Double = if (value > 0) 1.0 else 0.0
        override fun apply(value: Double): Double = Math.max(value, 0.0)
    };

    abstract fun apply(value: Double): Double
    abstract fun derivative(value: Double): Double
}