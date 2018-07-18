package cz.maks.strategies

import com.fasterxml.jackson.annotation.JsonSubTypes
import com.fasterxml.jackson.annotation.JsonTypeInfo
import java.util.*

object BiasInitialisation {
    fun zeros(): BiasInitialisationFunction = ZerosInitializer()
    fun random(lowerBound: Double = -0.5, upperBound: Double = 0.7): BiasInitialisationFunction =
            RandomInitializer(lowerBound, upperBound)
}

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "type")
@JsonSubTypes(
        JsonSubTypes.Type(name = "Zeros", value = ZerosInitializer::class),
        JsonSubTypes.Type(name = "Random", value = RandomInitializer::class)
)
interface BiasInitialisationFunction {
    fun apply(rand: Random = Random()): Double
}

private class ZerosInitializer : BiasInitialisationFunction {
    override fun apply(rand: Random): Double = 0.0
}

private class RandomInitializer(
        val lowerBound: Double = -0.5,
        val upperBound: Double = 0.7
) : BiasInitialisationFunction {
    override fun apply(rand: Random): Double = RandomUtils.generateRandom(lowerBound, upperBound, rand)
}