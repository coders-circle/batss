__kernel void main(
    __global float* io_array,
    __global float* weight_array,
    __global float* delta_array,
    int layer_offset,
    int layer_size,
    int forward_layer_offset,
    int learning_rate)
{
    int ci = get_global_id(0);

    wi = weight_offset + layer_size*ci;
    float dwc = delta_array[layer_offset+ci]*learning_rate;
    for(int i = 0; i < layer_size; i++)
    {
        weight_array[wi + i] += dwc * io_array[forward_layer_offset+i];
    }
}
