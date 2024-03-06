#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdarg.h>
#include <time.h>

#define MAX_THREADS 8


int main()
{

    printf("Hello World!\n");

    printf("Using multiprocessing\n");
        // Initialize mutex
        pthread_mutex_t mutex;
        pthread_mutex_init(&mutex, NULL);

        // Create thread pool
        pthread_t threads[MAX_THREADS];
        int thread_count = 0;

        // Create threads for computing LNC
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                if (thread_count >= MAX_THREADS) {
                    // Wait for a thread to finish before starting a new one
                    pthread_join(threads[thread_count % MAX_THREADS], NULL);
                    thread_count++;
                }
                thread_count++;
                printf("thread %d running...\n", (thread_count % MAX_THREADS));
            }
        }

        // Join threads
        for (int i = 0; i < MAX_THREADS && i < thread_count; i++) {
            pthread_join(threads[i], NULL);
        }

        // Destroy mutex
        pthread_mutex_destroy(&mutex);

    return 0;
}
