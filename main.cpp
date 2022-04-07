#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "src/gsv_wrapper.h"

int char2int(char c) {
    if ('0' <= c && c <= '9')
        return c - '0';
    else if ('a' <= c && c <= 'f')
        return c - 'a' + 10;
    else if ('A' <= c && c <= 'F')
        return c - 'A' + 10;
    else {
        printf("Invalid char: '%c'\n", c);
        exit(1);
    }
}

void hex2bn(uint32_t *x, const char *hex_string, int cnt) {
    int index = 0, length = 0, value;

    for (index = 0; index < cnt; index++) x[index] = 0;

    while (hex_string[length] != 0) length++;

    for (index = 0; index < length; index++) {
        value = char2int(hex_string[length - index - 1]);
        x[index / 8] += value << index % 8 * 4;
    }
}

void print_bn(uint32_t *x, uint32_t cnt) {
    int index;

    for (index = cnt - 1; index >= 0; index--) {
        printf("%08X", x[index]);
    }
    printf("\n");
}

void test_sign(int num_gpus, int count) {
    gsv_sign_t *sig;

    sig = (gsv_sign_t *)malloc(sizeof(gsv_sign_t) * count);

    for (int i = 0; i < count; i++) {
        hex2bn(sig[i].e._limbs, "10D51CB90C0C0522E94875A2BEA7AB72299EBE7192E64EFE0573B1C77110E5C9", GSV_BITS / 32);
        hex2bn(sig[i].priv_key._limbs, "128B2FA8BD433C6C068C8D803DFF79792A519A55171B1B650C23661D15897263", GSV_BITS / 32);
        hex2bn(sig[i].k._limbs, "E11F5909F947D5BE08C84A22CE9F7C338F7CF4A5B941B9268025495D7D433071", GSV_BITS / 32);
    }

    GSV_sign_exec(num_gpus, count, sig);

    // for (int i = 0; i < 1; i++) {
    //     print_bn(sig[i].r._limbs, GSV_BITS / 32);
    //     print_bn(sig[i].s._limbs, GSV_BITS / 32);
    // }
}

void test_verify(int num_gpus, int count) {
    gsv_verify_t *sig;
    int *results;

    sig = (gsv_verify_t *)malloc(sizeof(gsv_verify_t) * count);
    results = (int *)malloc(sizeof(int) * count);

    for (int i = 0; i < count; i++) {
        hex2bn(sig[i].r._limbs, "23B20B796AAAFEAAA3F1592CB9B4A93D5A8D279843E1C57980E64E0ABC5F5B05", GSV_BITS / 32);
        hex2bn(sig[i].s._limbs, "E11F5909F947D5BE08C84A22CE9F7C338F7CF4A5B941B9268025495D7D433071", GSV_BITS / 32);
        hex2bn(sig[i].e._limbs, "10D51CB90C0C0522E94875A2BEA7AB72299EBE7192E64EFE0573B1C77110E5C9", GSV_BITS / 32);
#ifndef GSV_KNOWN_PKEY
        hex2bn(sig[i].key_x._limbs, "D5548C7825CBB56150A3506CD57464AF8A1AE0519DFAF3C58221DC810CAF28DD", GSV_BITS / 32);
        hex2bn(sig[i].key_y._limbs, "921073768FE3D59CE54E79A49445CF73FED23086537027264D168946D479533E", GSV_BITS / 32);
#endif
    }

    GSV_verify_exec(num_gpus, count, sig, results);

    for (int i = 0; i < count; i++) {
        if (results[i] != 0) {
            printf("Signature #%d does not match public key.\n", i);
            break;
        }
    }
}

int main(int argc, char **argv) {
    for (int num_gpus = 1; num_gpus <= 8; num_gpus++) {
        printf("#GPU %d\n", num_gpus);

        // Signature generation benchmark
        GSV_sign_init(num_gpus);

        for (int i = 256; i <= 8388608; i *= 2) {
            printf("#instances: %d\n", i);
            test_sign(num_gpus, i);
            test_sign(num_gpus, i);
            test_sign(num_gpus, i);
        }

        GSV_sign_close(num_gpus);

        // Signature verification benchmark
        GSV_verify_init(num_gpus);

        for (int i = 256; i <= 1048576; i *= 2) {
            printf("#instances: %d\n", i);
            test_verify(num_gpus, i);
            test_verify(num_gpus, i);
            test_verify(num_gpus, i);
        }

        GSV_verify_close(num_gpus);
    }

    return 0;
}
