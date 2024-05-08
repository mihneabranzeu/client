package com.federatedlearningplatform.client

import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.federatedlearningplatform.client.ui.theme.ClientTheme
import com.federatedlearningplatform.fl_tensorflowlite.FLClient
import com.federatedlearningplatform.fl_tensorflowlite.FLService
import com.federatedlearningplatform.fl_tensorflowlite.SampleSpec
import com.federatedlearningplatform.fl_tensorflowlite.createFLService
import com.federatedlearningplatform.fl_tensorflowlite.helpers.classifierAccuracy
import com.federatedlearningplatform.fl_tensorflowlite.helpers.loadMappedAssetFile
import com.federatedlearningplatform.fl_tensorflowlite.helpers.negativeLogLikelihoodLoss
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient

class MainActivity : ComponentActivity() {
    lateinit var flClient: FLClient<Float3DArray, FloatArray>
    lateinit var flService: FLService<Float3DArray, FloatArray>
    private val scope = MainScope()
    private val resultText = mutableStateOf("")
    private val deviceId = mutableStateOf("")
    private val isLoadButtonEnabled = mutableStateOf(true)
    private val isTrainButtonEnabled = mutableStateOf(false)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        createFLClient()
        doCreateFLService()
        setContent {
            ClientTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Column(modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally) {

                        TextField(value = deviceId.value,
                            onValueChange = {deviceId.value = it},
                            label = { Text("Device ID") },
                            modifier = Modifier.fillMaxWidth())

                        Button(onClick = { scope.launch { makeHttpRequest() } },
                            modifier = Modifier
                                .height(60.dp)
                                .padding(top = 16.dp),
                            enabled = isLoadButtonEnabled.value) {
                            Text("Load Data")
                        }
                        Button(onClick = { handleGetParameters() },
                            modifier = Modifier
                                .height(60.dp)
                                .padding(top = 16.dp),
                            enabled = isTrainButtonEnabled.value) {
                            Text("Get Parameters")
                        }
                        Button(onClick = { handleFit() },
                            modifier = Modifier
                                .height(60.dp)
                                .padding(top = 16.dp),
                            enabled = isTrainButtonEnabled.value) {
                            Text("Train")
                        }
                        Button(onClick = { handleEvaluate() },
                            modifier = Modifier
                                .height(60.dp)
                                .padding(top = 16.dp),
                            enabled = isTrainButtonEnabled.value) {
                            Text("Evaluate")
                        }

                        Spacer(modifier = Modifier.height(16.dp))

                        Box(modifier = Modifier
                            .fillMaxWidth()
                            .height(400.dp)
                            .border(2.dp, Color.Black)
                            ) {
                            Text(resultText.value)
                        }
                    }

                }
            }
        }
    }

    private fun handleGetParameters() {
       flService.getParameters()
    }

    private fun handleFit() {
        flService.fit()
    }

    private fun handleEvaluate() {
        flService.evaluate()
    }

    private suspend fun makeHttpRequest() = withContext(Dispatchers.IO) {
        val url = "http://localhost:5000/api/projects"
        val client = OkHttpClient()
        val request = okhttp3.Request.Builder()
            .url(url)
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw Error("Unexpected code $response")
        }
    }

    private fun createFLClient() {
        val buffer = loadMappedAssetFile(this, "model/cifar10.tflite")
        val layersSizes = intArrayOf(1800, 24, 9600, 64, 768000, 480, 40320, 336, 3360, 40)
        flClient = FLClient(buffer, layersSizes, SampleSpec(
            { it.toTypedArray() },
            { it.toTypedArray() },
            { Array(it) { FloatArray(CLASSES.size) } },
            ::negativeLogLikelihoodLoss,
            ::classifierAccuracy
        ))
    }

    fun doCreateFLService() {
        flService = createFLService(flClient) {
            runOnUiThread {
                setTextResult(it)
            }
        }
    }

    fun loadData(deviceId: String) {
        if (deviceId.isEmpty() || !(1 .. 10).contains(deviceId.toInt())) {
          Toast.makeText(this, "Please enter a client partition ID between 1 and 10 (inclusive)", Toast.LENGTH_LONG).show()
        } else {
            // Disable button
            isLoadButtonEnabled.value = false
            isTrainButtonEnabled.value = false
            setTextResult("Loading the local training dataset in memory. It will take several seconds.")
            scope.launch {
                loadDataInBackground()
            }
        }

    }

    suspend fun loadDataInBackground() {
        val result = runWithStacktraceOr("Failed to load training dataset.") {
            loadData(this, flClient, deviceId.value.toInt())
            "Training dataset is loaded in memory. Ready to train!\nTrain Size: ${flClient.trainingSamples.size}\n${flClient.trainingSamples[0].label.map { it.toString() }}"

        }
        runOnUiThread {
            setTextResult(result)
            isLoadButtonEnabled.value = true
            isTrainButtonEnabled.value = true
        }
    }

   suspend fun <T> runWithStacktraceOr(or: T, call: suspend () -> T): T {
        return try {
            call()
        } catch (err: Error) {
            Log.e(TAG, Log.getStackTraceString(err))
            or
        }
    }

    fun setTextResult(text: String) {
        resultText.value = text
    }
}

private const val TAG = "MainActivity"
typealias Float3DArray = Array<Array<FloatArray>>
