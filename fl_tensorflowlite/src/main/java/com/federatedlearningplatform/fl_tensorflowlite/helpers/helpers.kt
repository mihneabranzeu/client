package com.federatedlearningplatform.fl_tensorflowlite.helpers

import android.content.Context
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

fun loadMappedAssetFile(context: Context, filePath: String): MappedByteBuffer {
    val fileDescriptor = context.assets.openFd(filePath)
    val fileChannel = fileDescriptor.createInputStream().channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
}

fun FloatArray.argmax(): Int = indices.maxBy { this[it] }

infix fun <T, R> Iterable<T>.lazyZip(other: Array<out R>): Sequence<Pair<T, R>> {
    val ours = iterator()
    val theirs = other.iterator()

    return sequence {
        while (ours.hasNext() && theirs.hasNext()) {
            yield(ours.next() to theirs.next())
        }
    }
}

fun assertIntsEqual(expected: Int, actual: Int) {
    if (expected != actual) {
        throw AssertionError("Test failed: expected `$expected`, got `$actual` instead.")
    }
}