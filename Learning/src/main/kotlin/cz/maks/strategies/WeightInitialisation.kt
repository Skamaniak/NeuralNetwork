package cz.maks.strategies

import com.fasterxml.jackson.annotation.JsonSubTypes
import com.fasterxml.jackson.annotation.JsonTypeInfo
import java.util.*

object WeightInitialisation {
    fun xavierNormal(): WeightInitialisationFunction = XavierNormal()
    fun heNormal(): WeightInitialisationFunction = HeNormal()
    fun leCun(): WeightInitialisationFunction = LeCunNormal()
    fun random(lowerBound: Double = -1.0, upperBound: Double = 1.0): WeightInitialisationFunction =
            RandomInit(lowerBound, upperBound)
}

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "type")
@JsonSubTypes(
        JsonSubTypes.Type(name = "XavierNormal", value = XavierNormal::class),
        JsonSubTypes.Type(name = "HeNormal", value = HeNormal::class),
        JsonSubTypes.Type(name = "LeCunNormal", value = LeCunNormal::class),
        JsonSubTypes.Type(name = "Random", value = RandomInit::class)
)
interface WeightInitialisationFunction {
    fun apply(inCount: Int, outCount: Int, rand: Random = Random()): Double
}

private class XavierNormal : WeightInitialisationFunction {
    override fun apply(inCount: Int, outCount: Int, rand: Random): Double {
        val standardDeviation = Math.sqrt(2 / (inCount.toDouble() + outCount.toDouble()))
        return rand.nextGaussian() * standardDeviation
    }
}

private class HeNormal : WeightInitialisationFunction {
    override fun apply(inCount: Int, outCount: Int, rand: Random): Double {
        val standardDeviation = Math.sqrt(2 / inCount.toDouble())
        return rand.nextGaussian() * standardDeviation
    }
}

private class LeCunNormal : WeightInitialisationFunction {
    override fun apply(inCount: Int, outCount: Int, rand: Random): Double {
        val standardDeviation = Math.sqrt(1 / inCount.toDouble())
        return rand.nextGaussian() * standardDeviation
    }
}

private class RandomInit(
        val lowerBound: Double = -1.0,
        val upperBound: Double = 1.0
) : WeightInitialisationFunction {
    override fun apply(inCount: Int, outCount: Int, rand: Random): Double =
            RandomUtils.generateRandom(lowerBound, upperBound, rand)
}