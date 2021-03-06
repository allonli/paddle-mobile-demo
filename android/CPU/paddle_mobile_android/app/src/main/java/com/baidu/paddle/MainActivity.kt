/*
 * Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of
 * the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
package com.baidu.paddle

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.support.v4.content.FileProvider
import android.text.InputType
import android.widget.Toast
import com.afollestad.materialdialogs.MaterialDialog
import com.baidu.paddle.data.banana
import com.baidu.paddle.data.tempImage
import com.baidu.paddle.modeloader.LoaderFactory
import com.baidu.paddle.modeloader.ModelLoader
import com.baidu.paddle.modeloader.ModelType
import com.baidu.paddle.utils.FileUtils
import com.baidu.paddle.utils.PermissionUtils
import io.reactivex.Observable
import io.reactivex.ObservableEmitter
import io.reactivex.Observer
import io.reactivex.android.schedulers.AndroidSchedulers
import io.reactivex.disposables.Disposable
import io.reactivex.schedulers.Schedulers
import kotlinx.android.synthetic.main.main_activity.*
import org.jetbrains.anko.AnkoLogger
import org.jetbrains.anko.info
import java.io.File


@SuppressLint("SetTextI18n")
class MainActivity : Activity(), AnkoLogger {

    private lateinit var mModelLoader: ModelLoader

    private var mThreadCounts = 1
    private var mPredictCounts = 1L
    private val modelList: ArrayList<ModelType> by lazy {
        val list = ArrayList<ModelType>()
        list.add(ModelType.mobilenet)
        list
    }
    private var mCurrentType = modelList[0]


    private val threadCountList: ArrayList<Int> by lazy {
        Runtime.getRuntime().availableProcessors()
        val list = ArrayList<Int>()
        for (i in (1..Runtime.getRuntime().availableProcessors() / 2)) {
            list.add(i)
        }
        list
    }


    private var isloaded = false
    private var isModelCopyed = false
    private var mCurrentPath: String? = banana.absolutePath
    private val isGotNeededPermissions: Boolean
        get() = PermissionUtils.checkPermissions(this, Manifest.permission.CAMERA) && PermissionUtils.checkPermissions(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)


    /**
     * check whether sdcard is mounted
     */
    private val isHasSdCard: Boolean
        get() = Environment.getExternalStorageState() == Environment.MEDIA_MOUNTED


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main_activity)
        init()
        if (!isGotNeededPermissions) {
            doRequestPermission()
        } else {
            copyModels()
        }
    }

    private fun doRequestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val permissions = arrayOf(Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE)
            this.requestPermissions(permissions, PERMISSION_REQUEST_CODE)
        }
    }

    private fun init() {
        updateCurrentModel()
        mModelLoader.setThreadCount(mThreadCounts)
        thread_counts.text = "$mThreadCounts"
        clearInfos()
        mCurrentPath = banana.absolutePath
        predict_banada.setOnClickListener {
            scaleImageAndPredictImage(mCurrentPath, mPredictCounts)
        }
        btn_takephoto.setOnClickListener {
            if (!isHasSdCard) {
                Toast.makeText(this@MainActivity, R.string.sdcard_not_available,
                        Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }
            takePicFromCamera()

        }
        bt_load.setOnClickListener {
            isloaded = true
            mModelLoader.load()
        }

        bt_clear.setOnClickListener {
            isloaded = false
            mModelLoader.clear()
            clearInfos()
        }
        ll_model.setOnClickListener {
            MaterialDialog.Builder(this)
                    .title("选择模型")
                    .items(modelList)
                    .itemsCallbackSingleChoice(modelList.indexOf(mCurrentType))
                    { _, _, which, text ->
                        info { "which=$which" }
                        info { "text=$text" }
                        mCurrentType = modelList[which]
                        updateCurrentModel()
                        reloadModel()
                        clearInfos()
                        true
                    }
                    .positiveText("确定")
                    .show()
        }

        ll_threadcount.setOnClickListener {
            MaterialDialog.Builder(this)
                    .title("设置线程数量")
                    .items(threadCountList)
                    .itemsCallbackSingleChoice(threadCountList.indexOf(mThreadCounts))
                    { _, _, which, _ ->
                        mThreadCounts = threadCountList[which]
                        info { "mThreadCounts=$mThreadCounts" }
                        mModelLoader.setThreadCount(mThreadCounts)
                        reloadModel()
                        thread_counts.text = "$mThreadCounts"
                        clearInfos()
                        true
                    }
                    .positiveText("确定")
                    .show()
        }

        runcount_counts.text = "$mPredictCounts"

        ll_runcount.setOnClickListener {
            MaterialDialog.Builder(this)
                    .inputType(InputType.TYPE_CLASS_NUMBER)
                    .input("设置预测次数", "10") { _, input ->
                        mPredictCounts = input.toString().toLong()
                        info { "mRunCount=$mPredictCounts" }
                        mModelLoader.mTimes = mPredictCounts
                        reloadModel()
                        runcount_counts.text = "$mPredictCounts"
                    }.inputRange(1, 3)
                    .show()
        }
    }

    private fun reloadModel() {
        mModelLoader.clear()
        mModelLoader.load()
        isloaded = true
    }

    private fun updateCurrentModel() {
        tv_modetext.text = mCurrentType.name
        mModelLoader = LoaderFactory.buildLoader(mCurrentType)
    }

    private fun takePicFromCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        val path = tempImage.path
        mCurrentPath = path
        val mOriginUri: Uri
        mOriginUri = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            FileProvider.getUriForFile(this@MainActivity.application, this@MainActivity.application.packageName + ".FileProvider",
                    File(path))
        } else {
            Uri.fromFile(File(path))
        }
        intent.putExtra(MediaStore.EXTRA_OUTPUT, mOriginUri)

        this@MainActivity.startActivityForResult(intent, TAKE_PHOTO_REQUEST_CODE)
    }

    @SuppressLint("CheckResult")
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (isGotNeededPermissions) {
                copyModels()
            } else {
                doRequestPermission()
            }
        }
    }

    @SuppressLint("CheckResult", "SetTextI18n")
    private fun copyModels() {
        if (isModelCopyed) {
            return
        }
        tv_infos?.text = "拷贝模型中...."
        val dialog = MaterialDialog.Builder(this)
                .title("模型拷贝中")
                .content("请稍等..")
                .progress(true, 0)
                .show()

        Observable.create { emitter: ObservableEmitter<String> ->
            val assetPath = "pml_demo"
            val sdcardPath = (Environment.getExternalStorageDirectory().toString() + File.separator + assetPath)
            FileUtils.delDir(sdcardPath)
            FileUtils.copyFilesFromAssets(this@MainActivity, assetPath, sdcardPath)
            emitter.onNext(sdcardPath)
        }
                .subscribeOn(Schedulers.io())
                .observeOn(AndroidSchedulers.mainThread(), true)
                .subscribe { path ->
                    isModelCopyed = true
                    tv_infos.text = "模型已拷贝至$path"
                    scaleAndShowBitmap(banana.absolutePath)
                    dialog.dismiss()
                }
    }


    public override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        when (requestCode) {
            TAKE_PHOTO_REQUEST_CODE -> if (resultCode == Activity.RESULT_OK) {
                scaleAndShowBitmap(mCurrentPath)
            }
            else -> {
            }
        }
    }

    private fun scaleAndShowBitmap(path: String?) {
        Observable
                .just(path)
                .map {
                    mModelLoader.getScaleBitmap(
                            this@MainActivity,
                            this.mCurrentPath
                    )
                }
                .subscribeOn(Schedulers.io())
                .observeOn(AndroidSchedulers.mainThread())
                .doOnNext { bitmap -> show_image.setImageBitmap(bitmap) }
                .subscribe()
    }

    /**
     * 缩放然后predict这张图片
     */
    private fun scaleImageAndPredictImage(path: String?, times: Long) {
        if (path == null) {
            Toast.makeText(this, "图片lost", Toast.LENGTH_SHORT).show()
            return
        }
        if (mModelLoader.isbusy) {
            Toast.makeText(this, "处于前一次操作中", Toast.LENGTH_SHORT).show()
            return
        }
        mModelLoader.clearTimeList()
        tv_infos.text = "预处理数据,执行运算..."
        mModelLoader.predictTimes(times)
        Observable
                .just(path)
                .map {
                    if (!isloaded) {
                        isloaded = true
                        mModelLoader.setThreadCount(mThreadCounts)
                        mModelLoader.load()
                    }
                    mModelLoader.getScaleBitmap(
                            this@MainActivity,
                            path
                    )
                }
                .subscribeOn(Schedulers.io())
                .observeOn(AndroidSchedulers.mainThread())
                .doOnNext { bitmap -> show_image.setImageBitmap(bitmap) }
                .map { bitmap ->
                    var floatsTen: FloatArray? = null
                    for (i in 0..(times - 1)) {
                        val floats = mModelLoader.predictImage(bitmap)
                        val predictImageTime = mModelLoader.predictImageTime
                        mModelLoader.timeList.add(predictImageTime)
                        if (i == times / 2) {
                            floatsTen = floats
                        }
                    }
                    Pair(floatsTen!!, bitmap)
                }
                .observeOn(AndroidSchedulers.mainThread())
                .map { floatArrayBitmapPair ->
                    mModelLoader.mixResult(show_image, floatArrayBitmapPair)
                    floatArrayBitmapPair.second
                    floatArrayBitmapPair.first
                }
                .observeOn(Schedulers.io())
                .map(mModelLoader::processInfo)
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(object : Observer<String?> {
                    override fun onSubscribe(d: Disposable) {
                        mModelLoader.isbusy = true
                    }

                    override fun onNext(resultInfo: String) {
                        tv_infomain.text = mModelLoader.getMainMsg()
                        tv_preinfos.text =
                                mModelLoader.getDebugInfo() + "\n" +
                                        mModelLoader.timeInfo + "\n" +
                                        "点击查看结果"

                        tv_preinfos.setOnClickListener {
                            MaterialDialog.Builder(this@MainActivity)
                                    .title("结果:")
                                    .content(resultInfo)
                                    .show()
                        }
                    }

                    override fun onComplete() {
                        mModelLoader.isbusy = false
                        tv_infos.text = ""
                    }

                    override fun onError(e: Throwable) {
                        mModelLoader.isbusy = false
                    }
                })
    }

    private fun clearInfos() {
        tv_infos.text = ""
        tv_preinfos.text = ""
    }

    override fun onBackPressed() {
        super.onBackPressed()

        info { "pml clear" }
        // clear pml
        isloaded = false
        mModelLoader.clear()
    }

    companion object {
        internal const val TAG = "PML"
        const val TAKE_PHOTO_REQUEST_CODE = 1001
        const val PERMISSION_REQUEST_CODE = 1002

        init {
            try {
                System.loadLibrary("paddle-mobile")
            } catch (e: SecurityException) {
                e.printStackTrace()
            } catch (e: UnsatisfiedLinkError) {
                e.printStackTrace()
            } catch (e: NullPointerException) {
                e.printStackTrace()
            }
        }
    }
}
