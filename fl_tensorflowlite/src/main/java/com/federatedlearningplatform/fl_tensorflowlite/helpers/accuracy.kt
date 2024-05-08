package com.federatedlearningplatform.fl_tensorflowlite.helpers

import com.federatedlearningplatform.fl_tensorflowlite.Sample

fun <X> classifierAccuracy(
    samples: MutableList<Sample<X, FloatArray>>,
    logits: Array<FloatArray>
): Float = averageLossWith(samples, logits) { sample, logit ->
    sample.label[logit.argmax()]
}

/** Signifies that "accuracy" makes no sense for the model. */
@Suppress("UNUSED_PARAMETER")
fun <X, Y> placeholderAccuracy(
    samples: MutableList<Sample<X, Y>>,
    logits: Array<Y>
): Float = -1f
