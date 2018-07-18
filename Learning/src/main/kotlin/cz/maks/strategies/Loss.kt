package cz.maks.strategies

import com.fasterxml.jackson.annotation.JsonSubTypes
import com.fasterxml.jackson.annotation.JsonTypeInfo

object Loss {
    fun difference(): LossFunction = Difference()
}

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "type")
@JsonSubTypes(
        JsonSubTypes.Type(name = "Difference", value = Difference::class)
)
interface LossFunction {
    fun apply(output: Double, target: Double): Double
}

private class Difference : LossFunction {
    override fun apply(output: Double, target: Double): Double = (output - target)
}