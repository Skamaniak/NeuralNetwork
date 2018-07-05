package cz.maks.persistence

import com.fasterxml.jackson.databind.ObjectMapper
import cz.maks.ex.MalformedNetworkArchiveException
import cz.maks.model.NeuralNetwork
import cz.maks.model.NeuralNetworkDefinition
import cz.maks.model.SerialisationModel
import java.io.File
import java.nio.charset.StandardCharsets
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

/**
 * Created by Jan Skrabal skrabalja@gmail.com
 */

object FilePersistence {
    private val MAPPER = ObjectMapper()
    private val ENTRY_NAME = "network.json"

    fun store(network: NeuralNetwork, path: String) {
        store(network, File(path))
    }

    fun store(network: NeuralNetwork, file: File) {
        val model = SerialisationModel.fromComputationalModel(network)
        val json = MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(model)
        file.writeText(json)

        file.outputStream().use {
            ZipOutputStream(it, StandardCharsets.UTF_8).use {
                it.putNextEntry(ZipEntry(ENTRY_NAME))
                it.bufferedWriter(StandardCharsets.UTF_8).use {
                    it.write(json)
                }
            }
        }
    }

    fun load(path: String): NeuralNetwork {
        return load(File(path))
    }

    fun load(file: File): NeuralNetwork {
        val json = file.inputStream().use {
            ZipInputStream(it, StandardCharsets.UTF_8). use{
                val found = seekNetworkEntry(it)
                if (!found) throw MalformedNetworkArchiveException("Zip entry $ENTRY_NAME not found in archive $file")
                it.bufferedReader(StandardCharsets.UTF_8).use {
                    it.readText()
                }
            }
        }
        val model = MAPPER.readValue(json, NeuralNetworkDefinition::class.java)
        return SerialisationModel.toComputationalModel(model)
    }

    private fun seekNetworkEntry(stream: ZipInputStream): Boolean {
        var entry = stream.nextEntry
        while (entry != null && entry.name != ENTRY_NAME) {
            stream.closeEntry()
            entry = stream.nextEntry
        }
        return entry != null
    }
}