__kernel void main(
    __global float* io_array,
    __global float* weight_array,
    __global float* delta_array,
    int layer_offset,
    int learning_rate)
{
    int ci = get_global_id(0);
    int i = layer_offset+ci;
    weight_array[i] += learning_rate*delta_array[i]*io_array[i];
}
