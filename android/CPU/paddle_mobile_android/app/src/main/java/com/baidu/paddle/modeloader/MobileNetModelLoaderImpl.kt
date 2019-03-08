package com.baidu.paddle.modeloader

import android.graphics.Bitmap
import android.os.Environment
import com.baidu.paddle.PML
import com.baidu.paddle.data.MobileNetClassfiedData
import org.jetbrains.anko.info
import java.io.File


/**
 * a sample for mobilenet classification
 *
 * Created by xiebaiyuan on 2018/7/18.
 */
class MobileNetModelLoaderImpl : ModelLoader() {

    /* b g r
     mean_value: [103.94,116.78,123.68]

     transform_param

     {
         scale: 0.017
         mirror: false
         crop_size: 224
         mean_value: [103.94, 116.78, 123.68]
     }

     input_dim: 1
     input_dim: 3
     input_dim: 224
     input_dim: 224*/


    private var type = ModelType.mobilenet
    // mobile net is bgr
    private val means = floatArrayOf(103.94f, 116.78f, 123.68f)
    private val ddims = intArrayOf(1, 3, 224, 224)
    private val scale = 0.017f
    private var mResultInfo: String? = ""
    private var mDebugInfo: String? = ""

    override fun getScaledMatrix(bitmap: Bitmap, desWidth: Int, desHeight: Int): FloatArray {
        val rsGsBs = getRsGsBs(bitmap, desWidth, desHeight)

        val rs = rsGsBs.first
        val gs = rsGsBs.second
        val bs = rsGsBs.third

        val dataBuf = FloatArray(3 * desWidth * desHeight)

        if (rs.size + gs.size + bs.size != dataBuf.size) {
            throw IllegalArgumentException("rs.size + gs.size + bs.size != dataBuf.size should equal")
        }

        // bbbb... gggg.... rrrr...
        for (i in dataBuf.indices) {
            dataBuf[i] = when {
                i < bs.size -> (bs[i] - means[0]) * scale
                i < bs.size + gs.size -> (gs[i - bs.size] - means[1]) * scale
                else -> (rs[i - bs.size - gs.size] - means[2]) * scale
            }
        }

        return dataBuf
    }

    override fun getInputSize(): Int {
        return ddims[2]
    }

    override fun clear() {
        PML.clear()
    }

    override fun load() {
        val assetPath = "pml_demo"
        val sdcardPath = (Environment.getExternalStorageDirectory().toString()
                + File.separator + assetPath + File.separator + type)
        PML.load(sdcardPath,false)
    }

    override fun predictImage(inputBuf: FloatArray): FloatArray? {
        var predictImage: FloatArray? = null
        try {
            val start = System.currentTimeMillis()
            predictImage = PML.predictImage(inputBuf, ddims)
            val end = System.currentTimeMillis()
            predictImageTime = end - start
        } catch (e: Exception) {
        }
        return predictImage
    }

    override fun predictImage(bitmap: Bitmap): FloatArray? {
        return predictImage(getScaledMatrix(bitmap, getInputSize(), getInputSize()))
    }

    override fun processInfo(result: FloatArray): String? {

        val resultInfos = StringBuilder()

        var max = java.lang.Float.MIN_VALUE
        var maxi = -1
        var sum = 0f

        for (i in result.indices) {
            info { " index: " + i + " value: " + result[i] }
            sum += result[i]
            if (result[i] > max) {
                max = result[i]
                maxi = i
            }
        }
        resultInfos.appendln(" I am ${MobileNetClassfiedData.dataList[maxi]}")
        resultInfos.appendln()


        mResultInfo = resultInfos.toString()
        resultInfos.appendln(" maxindex: $maxi  max: $max")

        info { "maxindex: $maxi" }
        info { "max: $max" }
        info { "sum: $sum" }

        for (i in 0..Math.min(1000, result.size - 1)) {
            info { " index: " + i + " value: " + result[i] }
            resultInfos.appendln(" index: $i value: ${result[i]}")
            sum += result[i]
            if (result[i] > max) {
                max = result[i]
                maxi = i
            }
        }

        mDebugInfo = resultInfos.toString()
        return super.processInfo(result)
    }

    override fun getMainMsg(): String? {
        return mResultInfo
    }

    override fun getDebugInfo(): String? {
        return mDebugInfo
    }
}