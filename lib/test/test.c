#include "gpt_bpe.h"
#include "library.h"
#include <stdio.h>
#include <sys/mman.h>
#include <sys/fcntl.h>
#include <assert.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <time.h>


const int NANO_SECONDS_IN_SEC = 1000000000;
/* returns a static buffer of struct timespec with the time difference of ts1
 * and ts2, ts1 is assumed to be greater than ts2 */
struct timespec *TimeSpecDiff(struct timespec *ts1, struct timespec *ts2)
{
    static struct timespec ts;
    ts.tv_sec = ts1->tv_sec - ts2->tv_sec;
    ts.tv_nsec = ts1->tv_nsec - ts2->tv_nsec;
    if (ts.tv_nsec < 0) {
        ts.tv_sec--;
        ts.tv_nsec += NANO_SECONDS_IN_SEC;
    }
    return &ts;
}

char *testString = "This is a test string";

size_t mmap_file(char *path, char **mmap_addr) {
    int fd = open(path, O_RDONLY);
    assert (fd != -1);

    struct stat file_info;
    assert (fstat(fd, &file_info) != -1);

    *mmap_addr = mmap(NULL, file_info.st_size, PROT_READ, MAP_PRIVATE,
                     fd, 0);
    assert (mmap_addr != MAP_FAILED);
    return file_info.st_size;
}

size_t read_file(char *path, char **buf_ptr) {
    size_t bufsize = 0;
    FILE *fp = fopen(path, "r");
    if (fp != NULL) {
        /* Go to the end of the file. */
        if (fseek(fp, 0L, SEEK_END) == 0) {
            /* Get the size of the file. */
            bufsize = ftell(fp);
            if (bufsize == -1) { /* Error */ }

            /* Allocate our buffer to that size. */
            *buf_ptr = malloc(sizeof(char) * (bufsize + 1));

            /* Go back to the start of the file. */
            if (fseek(fp, 0L, SEEK_SET) != 0) { /* Error */ }

            /* Read the entire file into memory. */
            size_t newLen = fread(*buf_ptr, sizeof(char),
                                  bufsize, fp);
            if ( ferror( fp ) != 0 ) {
                fputs("Error reading file", stderr);
            } else {
                (*buf_ptr)[newLen++] = '\0'; /* Just to be safe. */
            }
        }
        fclose(fp);
    }
    return bufsize;
}

void benchmark_tokenize(char *input, size_t size) {
    uint64_t start_rdtsc, end_rdtsc, host_cpu_ticks;
    double host_cpu_ns, host_cpu_us, host_cpu_s, tokens_per_s;
    struct timespec begints, endts;
    uint64_t begin = 0, end = 0;

    clock_gettime(CLOCK_MONOTONIC, &begints);
    Tokens result = tokenizeBuffer("gpt2-tokenizer", input,
                                   size);
    clock_gettime(CLOCK_MONOTONIC, &endts);
    struct timespec *tmpts = TimeSpecDiff(&endts, &begints);
    uint64_t nsecElapsed = (unsigned long) tmpts->tv_sec * \
                                1000000000LL + tmpts->tv_nsec;

    // Calculate rates
    host_cpu_s = (double) nsecElapsed / 1000000000;
    tokens_per_s = (double) result.len / host_cpu_s;
    free(result.tokens);

    printf("TOKENS: %lu in %0.2f seconds, %0.2f tokens/s\n", result.len,
           host_cpu_s, tokens_per_s);
}

void bench_loop(char *input, size_t size, size_t reps) {
    for (size_t i=0; i<reps; ++i) {
        benchmark_tokenize(input, size);
    }
}


int main() {
    initTokenizer("gpt2-tokenizer");

    char* input;
    printf("mmap:\n");
    size_t size = mmap_file("../../all.txt",
                           &input);
    bench_loop(input, size, 10);

    printf("read:\n");
    size = read_file("../../all.txt",
                     &input);
    bench_loop(input, size, 10);

    return 0;
}