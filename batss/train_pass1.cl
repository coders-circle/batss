__kernel void main(
    __global float* expected_op_array,
    __global float* io_array,
    __global float* delta_array,
    int layer_op_start)
{
    int i = layer_op_start;
    int ci = get_global_id(0);
    delta_array[ci + i] = expected_op_array[ci] - io_array[ci];
}
