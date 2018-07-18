package cz.maks.strategies

import com.fasterxml.jackson.annotation.JsonSubTypes
import com.fasterxml.jackson.annotation.JsonSubTypes.Type
import com.fasterxml.jackson.annotation.JsonTypeInfo
import kotlin.math.E
import kotlin.math.pow

object Activation {
    fun sigmoid(): ActivationFunction = SigmoidActivation()
    fun tanh(): ActivationFunction = TanhActivation()
    fun relu(): ActivationFunction = RectifiedLinearUnitActivation()
}

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "type")
@JsonSubTypes(
    Type(name = "Sigmoid", value = SigmoidActivation::class),
    Type(name = "Tanh", value = TanhActivation::class),
    Type(name = "ReLu", value = RectifiedLinearUnitActivation::class)
)
interface ActivationFunction {
    fun apply(value: Double): Double
    fun derivative(value: Double): Double
}

private class SigmoidActivation : ActivationFunction {
    override fun derivative(value: Double): Double = value * (1 - value)
    override fun apply(value: Double): Double = 1.0 / (1.0 + E.pow(-value))
}

private class TanhActivation : ActivationFunction {
    override fun derivative(value: Double): Double = 1 - Math.pow(Math.tanh(value), 2.0)
    override fun apply(value: Double): Double = Math.tanh(value)
}

private class RectifiedLinearUnitActivation : ActivationFunction {
    override fun derivative(value: Double): Double = if (value > 0) 1.0 else 0.0
    override fun apply(value: Double): Double = Math.max(value, 0.0)
}