package cz.maks

import cz.maks.persistence.FilePersistence

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
fun main(args: Array<String>) {
    NetworkTester("Digits#784-112-56-10#e65-s96,80.zip", DataSetType.TEST_DIGITS).test()
}

class NetworkTester(
        private val filePath: String,
        private val setType: DataSetType) {

    fun test() {
        val testSet = NetworkDataSetUtils.createTrainSet(setType)
        val network = FilePersistence.load(filePath)
        NetworkDataSetUtils.testTrainSet(network, testSet)
    }
}