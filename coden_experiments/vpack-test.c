#include <stdio.h>
#include <stdint.h>

void packing(const int8_t *A, const int8_t *B, int32_t *C, const int8_t *D, const int8_t *E) {
    __asm__ volatile(
        "vsetvli  t0, x0, e8, m1, tu, mu \n\t"
        "vle32.v  v0, (a0) \n\t"
        "add     a0, a0, 1 \n\t"
        "vle32.v  v1, (a0) \n\t"
        "add     a0, a0, 1 \n\t"
        "vle32.v  v2, (a0) \n\t"
        "add     a0, a0, 1 \n\t"
        "vle32.v  v3, (a0) \n\t"
        "add     a0, a0, 1 \n\t"
        "vsetvli  t0, x0, e64, m1, tu, mu \n\t"
        "vpack.vv v8, v0, v1, 0             \n\t"
        "vpack.vv v10, v2, v3, 0            \n\t"
        "vpack.vv v4, v8, v10, 0            \n\t"
        "vpack.vv v6, v9, v11, 0            \n\t" 
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
    packing(A, B, C, D, E);
    
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

