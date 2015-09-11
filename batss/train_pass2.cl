__kernel void main(
    __global float* expected_op_array,
    __global float* io_array,
    __global float* weight_array,
    __global float* pot_array,
    __global float* delta_array,
    int layer_offset,
    int forward_layer_offset,
    int weight_offset,
    int layer_size
    int forward_layer_size)
{
    int ci = get_global_id(0);
    int i = layer_offset;
    int lim = layer_offset + forward_layer_size;
    wi = weight_offset + layer_size*ci;
    float delta_temp = 0;
    for(int j = 0; i < lim; i++, j++)
    {
        delta_temp += delta_array[forward_layer_offset + ci + j]*weight_array[wi + j];
    }
    float f = 2/(1+exp(-pot_array[layer_offset+ci])) - 1;
    delta_array[layer_offset + ci] = delta_temp*0.5 * (1+f) * (1-f);
}
