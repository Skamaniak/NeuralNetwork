package cz.maks.strategies


/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
enum class LossFunction {
    DIFFERENCE {
        override fun apply(output: Double, target: Double): Double {
            return (output - target)
        }
    };

    abstract fun apply(output: Double, target: Double): Double
}