package cz.maks.train


/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */

class TrainSet(
        val inputCount: Int,
        val outputCount: Int,
        var data: Array<DataValue> = emptyArray()
) {

    fun addData(inputs: DoubleArray, outputs: DoubleArray): TrainSet {
        addData(DataValue(
                inputs = inputs,
                outputs = outputs
        ))
        return this
    }

    fun addData(dataValue: DataValue): TrainSet {
        validateDataEntry(dataValue)
        data = data.plus(dataValue)
        return this
    }

    fun extractSubset(size: Int): TrainSet {
        val randoms = exclusiveRandomVector(0, data.size, size)

        val subSet = TrainSet(inputCount, outputCount)
        randoms.forEach {
            subSet.addData(data[it])
        }
        return subSet
    }

    companion object {
        fun exclusiveRandomVector(lowerBound: Int, upperBound: Int, amount: Int): List<Int> {
            val range = lowerBound until upperBound
            return range.shuffled().take(amount)
        }
    }

    private fun validateDataEntry(dataValue: DataValue) {
        val dataInputCount = dataValue.inputs.size
        if (dataInputCount != inputCount) {
            throw IllegalArgumentException("Provided data has different number of inputs $dataInputCount than train " +
                    "set $inputCount")
        }

        val dataOutputCount = dataValue.outputs.size
        if (dataOutputCount != outputCount) {
            throw IllegalArgumentException("Provided data has different number of outputs $dataOutputCount than train " +
                    "set $outputCount")
        }
    }
}

data class DataValue(
        val inputs: DoubleArray,
        val outputs: DoubleArray
) {
    val inputCount = inputs.size
    val outputCount = outputs.size
}