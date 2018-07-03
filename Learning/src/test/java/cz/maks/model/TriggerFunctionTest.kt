package cz.maks.model

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.CsvSource

@DisplayName("Trigger function tests")
class TriggerFunctionTest {

    @DisplayName("Sigma function should return correct values")
    @ParameterizedTest(name = "function should return ''{1}'' for input ''{0}''")
    @CsvSource(
            "0, 0.5",
            Int.MIN_VALUE.toString() + ", 0",
            Int.MAX_VALUE.toString() + ", 1"

    )
    fun testSigmaImplementation(input: Double, expected: Double) {
        assertThat(TriggerFunction.SIGMOID.apply(input))
                .isEqualTo(expected)
    }


}
