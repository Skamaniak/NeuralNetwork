package cz.maks

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */
fun main(args: Array<String>) {
    val inputs = 28 * 28
    val outputs = 26
    val trainSetSize = 62400
    val scalingFactor = 2.0

    //https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    val numberOfHiddenNeurons = trainSetSize.toDouble() / (scalingFactor * (inputs.toDouble() + outputs.toDouble()))

    println("Number of neurons needed $numberOfHiddenNeurons")
}