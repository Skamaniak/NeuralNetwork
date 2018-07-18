package cz.maks.strategies

import java.util.*


internal object RandomUtils {
    private val RANDOM = Random()

    fun generateRandom(lower: Double, upper: Double, random: Random = RANDOM): Double {
        return random.nextDouble() * (upper - lower) + lower
    }
}