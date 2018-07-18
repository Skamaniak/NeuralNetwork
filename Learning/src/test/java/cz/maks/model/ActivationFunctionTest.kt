package cz.maks.model

import cz.maks.strategies.Activation
import cz.maks.strategies.ActivationFunction
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
        assertThat(Activation.sigmoid().apply(input))
                .isEqualTo(expected)
    }

    @DisplayName("TanH function should return correct values")
    @ParameterizedTest(name = "function should return ''{1}'' for input ''{0}''")
    @CsvSource(
            Int.MIN_VALUE.toString() + ", -1",
            "-0.5, -0.46211715726000974",
            "0, 0",
            "0.5, 0.46211715726000974",
            Int.MAX_VALUE.toString() + ", 1"

    )
    fun testTanHImplementation(input: Double, expected: Double) {
        assertThat(Activation.tanh().apply(input))
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
        assertThat(Activation.relu().apply(input))
                .isEqualTo(expected)
    }


}
