package cz.maks.model

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.CsvSource

@DisplayName("Trigger function tests")
class ActivationFunctionTest {

    @DisplayName("Sigma function should return correct values")
    @ParameterizedTest(name = "function should return ''{1}'' for input ''{0}''")
    @CsvSource(
            Int.MIN_VALUE.toString() + ", 0",
            "0, 0.5",
            Int.MAX_VALUE.toString() + ", 1"

    )
    fun testSigmaImplementation(input: Double, expected: Double) {
        assertThat(ActivationFunction.SIGMOID.apply(input))
                .isEqualTo(expected)
    }

    @DisplayName("TanH function should return correct values")
    @ParameterizedTest(name = "function should return ''{1}'' for input ''{0}''")
    @CsvSource(
            Int.MIN_VALUE.toString() + ", -1",
            "0, 0",
            Int.MAX_VALUE.toString() + ", 1"

    )
    fun testTanHImplementation(input: Double, expected: Double) {
        assertThat(ActivationFunction.TANH.apply(input))
                .isEqualTo(expected)
    }

    @DisplayName("ReLu function should return correct values")
    @ParameterizedTest(name = "function should return ''{1}'' for input ''{0}''")
    @CsvSource(
            Int.MIN_VALUE.toString() + ", 0",
            "-0.5, 0",
            "0, 0",
            "0.5, 0.5",
            "5, 5",
            "10, 10",
            Int.MAX_VALUE.toString() + "," + Int.MAX_VALUE.toString()

    )
    fun testReLuImplementation(input: Double, expected: Double) {
        assertThat(ActivationFunction.RELU.apply(input))
                .isEqualTo(expected)
    }

    @DisplayName("Leaky ReLu function should return correct values")
    @ParameterizedTest(name = "function should return ''{1}'' for input ''{0}''")
    @CsvSource(
            "-10, -0.1",
            "-5, -0.05",
            "-0.5, -0.005",
            "0, 0",
            "0.5, 0.5",
            "5, 5",
            "10, 10",
            Int.MAX_VALUE.toString() + "," + Int.MAX_VALUE.toString()

    )
    fun testLeakyReLuImplementation(input: Double, expected: Double) {
        assertThat(ActivationFunction.LEAKY_RELU.apply(input))
                .isEqualTo(expected)
    }

}
