__kernel void main(
    __global float* io_array,
    __global float* weight_array,
    __global float* delta_array,
    int layer_offset,
    int layer_size,
    int forward_layer_offset,
    int weight_offset,
    float learning_rate)
{
    int ci = get_global_id(0);

    int wi = weight_offset + layer_size*ci;
    for(int i = 0; i < layer_size; i++)
    {
        weight_array[wi + i] += learning_rate*delta_array[forward_layer_offset+i] * io_array[forward_layer_offset+i];
    }
}
