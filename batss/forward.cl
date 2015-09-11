__kernel void main(
    __global float* io_array,
    __global float* weight_array,
    __global float* pot_array,
    int input_offset,
    int weight_offset,
    int layer_size,
    int prev_layer_size)
{
    int ci = get_global_id(0);
    int i = input_offset;
    int lim = input_offset + layer_size;

    for (int j = 0; i < lim; i++, j++)
    {
        pot_array[input_offset+ci] += io_array[i] * weight_array[weight_offset + layer_size*j + ci];
    }

    io_array[input_offset+ci] = 2/(1+exp(-pot_array[input_offset+ci])) - 1;
    barrier(CLK_GLOBAL_MEM_FENCE);
}
