//step1:创建runtime
IRuntime* runtime = createInferRuntime(gLogger);
assert(runtime != nullptr);
//step2:反序列化创建engine
ICudaEngine* engine = runtime->deserializeCudaEngine(modelData, modelSize, nullptr);
assert(engine != nullptr);
// 打印绑定输入输出
printf("Bindings after deserializing:\n");
for (int bi = 0; bi < engine->getNbBindings(); bi++) 
{
    if (engine->bindingIsInput(bi) == true) 
    {
        printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
    } 
    else 
    {
        printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
    }
}



//step3:创建context,创建一些空间来存储中间激活值
IExecutionContext *context = engine->createExecutionContext();
assert(context != nullptr);

//step4:根据输入输出blob名字获取输入输出索引
int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
//step5:使用这些索引，创建buffers指向 GPU 上输入和输出缓冲区
void* buffers[2];
buffers[inputIndex] = inputBuffer;
buffers[outputIndex] = outputBuffer;
//step6：为输入输出开辟GPU显存
CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * inputDim.c() * inputDim.h() * inputDim.w() * sizeof(float)));
CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * outputDim.c() * outputDim.h() * outputDim.w() * sizeof(float)));
//step6：创建cuda流
cudaStream_t stream;
CUDA_CHECK(cudaStreamCreate(&stream));
//step7：从CPU到GPU----拷贝input数据
CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex],//显存上的存储区域，用于存放输入数据
                           input, //读入内存中的数据
                           batchSize * inputDim.c() * inputDim.h() * inputDim.w() * sizeof(float),
                           cudaMemcpyHostToDevice, 
                           stream));
//step8:异步推理
context->enqueueV2(buffers, stream, nullptr);

//step9：从GPU到CPU----拷贝output数据
CUDA_CHECK(cudaMemcpyAsync(output,//是内存中的数据
                           buffers[outputIndex],//是显存中的存储区,存放模型输出
                           batchSize * outputDim.c() * outputDim.h() * outputDim.w() * sizeof(float),
                           cudaMemcpyDeviceToHost,
                           stream));
//step10：同步cuda流
CUDA_CHECK(cudaStreamSynchronize(stream));
//step11：释放资源
cudaStreamDestroy(stream);
context->destroy();
engine->destroy();
runtime->destroy();
CUDA_CHECK(cudaFree(buffers[inputIndex]));
CUDA_CHECK(cudaFree(buffers[outputIndex]));

