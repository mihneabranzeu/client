package com.federatedlearningplatform.fl_tensorflowlite

import android.util.Log

class FLService<X: Any, Y: Any>(
    val flClient: FLClient<X, Y>,
    val callback: (String) -> Unit
){
    fun getParameters() {
       Log.d(TAG, "Handling GetParameters")
       callback("Handling GetParameters")
        val parameters = flClient.getParameters()
        Log.d(TAG, "Parameters: $parameters")
    }

    fun fit() {
        //TODO: Get parameters from the server and update the model
        Log.d(TAG, "Handling Fit")
        val epochs = 5
        callback("Handling Fit")
        flClient.fit(
            epochs,
            lossCallback = {callback("Avergae loss: ${it.average()}.")}
        )
    }

    fun evaluate() {
        Log.d(TAG, "Handling Evaluate")
        callback("Handling Evaluate")
        val (loss, accuracy) = flClient.evaluate()
        callback("Test Accuracy after this round = $accuracy")
    }

    companion object {
        private const val TAG = "FlService"
    }
}

fun <X:Any, Y: Any> createFLService(
    flClient: FLClient<X, Y>,
    callback: (String) -> Unit
): FLService<X, Y> {
    return FLService(flClient, callback)
}