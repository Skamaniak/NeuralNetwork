package cz.maks.strategies

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */


interface LearningAlgorythm {
    fun adjustStep(value: Double, derivative: Double)
}

private class Adam(
        private val alpha: Double = 0.001,
        private val beta1: Double = 0.9,
        private val beta2: Double = 0.999,
        private val epsilon: Double = Math.pow(10.0, -8.0)
) : LearningAlgorythm {
    override fun adjustStep(value: Double, derivative: Double) {

        // momentum - Exponentially weighted average
        val vdw = beta1

    }

}