// removed from the validate file since they aren't the proper approach
// keeping in case the code is helpful later.

// testing all possible combos for M0, N0, K0
    printf("flag, M0, N0, K0, K1, time\n");

    {    
        unsigned seed = 42;
        int flags[] = {0,1};
        int M0s[] = {4,8,16};
        int N0s[] = {4,8,16};
        // K0 should be the SEW, the size of the type.
    

        // loop thru flag options
        for (int f = 0; f < (int)(sizeof flags / sizeof flags[0]); f++) {
            // m0 loop
            for (int m = 0; m < (int)(sizeof M0s / sizeof M0s[0]); m++) {
                // n0 loop
                for (int n = 0; n < (int)(sizeof N0s / sizeof N0s[0]); n++) {
                    int M0 = M0s[m];
                    int N0 = N0s[n];
                    int K0 = 8;

                    // K1 = 8
                    int K1 = 8;
                    int8_t  lhs_buf8[K1 * M0 * K0];
                    int8_t  rhs_buf8[K1 * N0 * K0];
                    fill_random_s8(lhs_buf8, (size_t)(K1 * M0 * K0), &seed);
                    fill_random_s8(rhs_buf8, (size_t)(K1 * N0 * K0), &seed);
                    
                    clock_gettime(CLOCK_MONOTONIC, &start);
                    run_test(("test"), lhs_buf8, rhs_buf8, NULL, M0, N0, K0, K1, flags[f]);
                    clock_gettime(CLOCK_MONOTONIC, &end);
                    printf("%d, %d, %d, %d, %d, ", flags[f], M0, N0, K0, K1);
                    printf("%f\n", get_time_diff(start, end));

                    // K1 = 16
                    K1 = 16;
                    int8_t  lhs_buf16[K1 * M0 * K0];
                    int8_t  rhs_buf16[K1 * N0 * K0];
                    fill_random_s8(lhs_buf16, (size_t)(K1 * M0 * K0), &seed);
                    fill_random_s8(rhs_buf16, (size_t)(K1 * N0 * K0), &seed);
                    
                    clock_gettime(CLOCK_MONOTONIC, &start);
                    run_test(("test"), lhs_buf16, rhs_buf16, NULL, M0, N0, K0, K1, flags[f]);
                    clock_gettime(CLOCK_MONOTONIC, &end);
                    printf("%d, %d, %d, %d, %d, ", flags[f], M0, N0, K0, K1);
                    printf("%f\n", get_time_diff(start, end));

                    // K1 = 32
                    K1 = 32;
                    int8_t  lhs_buf32[K1 * M0 * K0];
                    int8_t  rhs_buf32[K1 * N0 * K0];
                    fill_random_s8(lhs_buf32, (size_t)(K1 * M0 * K0), &seed);
                    fill_random_s8(rhs_buf32, (size_t)(K1 * N0 * K0), &seed);
                    
                    clock_gettime(CLOCK_MONOTONIC, &start);
                    run_test(("test"), lhs_buf32, rhs_buf32, NULL, M0, N0, K0, K1, flags[f]);
                    clock_gettime(CLOCK_MONOTONIC, &end);
                    printf("%d, %d, %d, %d, %d, ", flags[f], M0, N0, K0, K1);
                    printf("%f\n", get_time_diff(start, end));
                }
                // printf("end m loop\n");
            }
            // printf("end f loop\n");
        } 
        // printf("end test\n");  
    }
    printf("done\n");