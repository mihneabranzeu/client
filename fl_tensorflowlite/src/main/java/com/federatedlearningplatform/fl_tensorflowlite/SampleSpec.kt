package com.federatedlearningplatform.fl_tensorflowlite

data class SampleSpec<X, Y>(
    val convertX: (List<X>) -> Array<X>,
    val convertY: (List<Y>) -> Array<Y>,
    val emptyY: (Int) -> Array<Y>,
    val loss: (MutableList<Sample<X, Y>>, Array<Y>) -> Float,
    val accuracy: (MutableList<Sample<X, Y>>, Array<Y>) -> Float,
)