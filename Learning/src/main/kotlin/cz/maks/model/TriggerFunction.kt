package cz.maks.model

import kotlin.math.E
import kotlin.math.pow


/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
enum class TriggerFunction {
    SIGMOID {
        override fun apply(sum: Double): Double {
            return 1.0 / (1.0 + E.pow(-sum))
        }
    };

    abstract fun apply(sum: Double): Double
}