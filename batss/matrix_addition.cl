__kernel void main(__global float* C, 
          __global float* A, 
          __global float* B)
{
    int i = get_global_id(0);
    C[i] = A[i] + B[i];
}