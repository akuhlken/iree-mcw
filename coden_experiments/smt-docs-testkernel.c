// A60 usage example:
// MatrixA[4, 8]          x  MatrixB[8, 4]    =   MatrixC[4, 4]
//    [0 1 2 3 4 5  6  7]       [0 1 2 11]           [140 168 196 224] 
//    [1 2 3 4 5 6  7  8]       [1 2 3  4]           [168 204 240 284] 
//    [2 3 4 5 6 7  8  9]       [2 3 4  5]           [196 240 284 344] 
//    [4 5 6 7 8 9 10 11]       [3 4 5  6]           [252 312 372 464] 
//                              [4 5 6  7]
//                              [5 6 7  8]
//                              [6 7 8  9]
//                              [7 8 9 10]

#include <stdio.h>
#include <stdint.h>

void matmul(const int8_t *A, const int8_t *B, int32_t *C, const int8_t *D, const int8_t *E) {
    __asm__ volatile(
        "vsetvli        t0, zero, e8, m1          \n\t"
        "vle8.v         v0, (%[A])                \n\t"

        "vle8.v         v1, (%[B])                \n\t"
        "smt.vmadot         v16, v0, v1               \n\t"
        "vsetvli        t0, zero, e32, m2         \n\t"
        "vse32.v        v16, (%[C])               \n\t"
        // "smt.vpack.vv       v8, v0              \n\t"
        "vupack.vv      v16, v7, v9       \n\t"
        "vse8.v         v7, (%[D])             \n\t"
        "vse8.v         v7, (%[E])             \n\t"
        : [ A ] "+r"(A), [ B ] "+r"(B), [ C ] "+r"(C), [ D ] "+r"(D), [ E ] "+r"(E)
        :
        : "cc");
}

int main()
{
    printf("Test Start.\n");
    // Init the matrixA, matrixB and matrixC.
    int8_t A[32] = {0, 1, 2, 3, 4, 5, 6, 7,
                    1, 2, 3, 4, 5, 6, 7, 8,
                    2, 3, 4, 5, 6, 7, 8, 9,
                    4, 5, 6, 7, 8, 9, 10, 11};
    int8_t B[32] = {0, 1, 2, 3, 4, 5, 6, 7,
                    1, 2, 3, 4, 5, 6, 7, 8,
                    2, 3, 4, 5, 6, 7, 8, 9,
                    11, 4, 5, 6, 7, 8, 9, 10};
    int32_t C[32] = {0, 0, 0, 0,
                     0, 0, 0, 0,
                     0, 0, 0, 0,
                     0, 0, 0, 0};

    int8_t D[32] = {0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0};
    int8_t E[32] = {0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0};
    
    // Call the FUNCTION
    matmul(A, B, C, D, E);
    
    // Print the OUTPUT
    for(int32_t iter_i=0; iter_i<4; iter_i++){
        for(int32_t iter_j=0; iter_j<4; iter_j++){
            printf("%d \t", C[iter_i*4 + iter_j]);
        }
        printf(" \n");
    }
    printf("Test End.\n");
    return 0;
}