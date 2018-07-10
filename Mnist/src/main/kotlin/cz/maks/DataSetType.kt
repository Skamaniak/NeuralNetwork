package cz.maks

enum class DataSetType(
        val imageFilePath: String,
        val labelFilePath: String,
        val outputs: Int,
        val inputs: Int = 28 * 28,
        val startIndex: Int = 0,
        val dataSetName: String = "unknown") {
    TRAIN_DIGITS(
            imageFilePath = "Mnist/src/main/resources/train/trainImage.idx3-ubyte",
            labelFilePath = "Mnist/src/main/resources/train/trainLabel.idx1-ubyte",
            outputs = 10,
            dataSetName = "Digits"
    ),
    TEST_DIGITS(
            imageFilePath = "Mnist/src/main/resources/test/t10k-images.idx3-ubyte",
            labelFilePath = "Mnist/src/main/resources/test/t10k-labels.idx1-ubyte",
            outputs = 10,
            dataSetName = "Digits"
    ),
    TRAIN_LETTERS(
            imageFilePath = "Mnist/src/main/resources/train/emnist-letters-train-images-idx3-ubyte",
            labelFilePath = "Mnist/src/main/resources/train/emnist-letters-train-labels-idx1-ubyte",
            outputs = 26,
            dataSetName = "Letters",
            startIndex = 1
    ),
    TEST_LETTERS(
            imageFilePath = "Mnist/src/main/resources/test/emnist-letters-test-images-idx3-ubyte",
            labelFilePath = "Mnist/src/main/resources/test/emnist-letters-test-labels-idx1-ubyte",
            outputs = 26,
            dataSetName = "Letters",
            startIndex = 1
    )
}