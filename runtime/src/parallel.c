#define _GNU_SOURCE

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <sys/sysinfo.h>

static int num_thread;

static atomic_bool flags[128];
static void (*fn)(void *, int, int);
static void *args;
static atomic_uint num_pending;

_Noreturn static void *thread_fn(void *p) {
  int i = (size_t) p, n = num_thread;
  while (1) {
    while (!flags[i]) pthread_yield();
    flags[i] = 0;
    fn(args, i, n);
    --num_pending;
  }
}

void parallel_init() __attribute__((constructor));

void parallel_init() {
  int n = num_thread = get_nprocs() / 2;
  cpu_set_t cpu;
  CPU_ZERO(&cpu);
  for (int i = 0; i < n; ++i) CPU_SET(i, &cpu);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu), &cpu); // master
  for (int i = 1; i < n; ++i) {
    pthread_t th;
    pthread_create(&th, 0, thread_fn, (void *) (size_t) i);
    CPU_ZERO(&cpu);
    CPU_SET(i, &cpu);
    pthread_setaffinity_np(th, sizeof(cpu), &cpu);
  }
}

void parallel_launch(void (*fn1)(void *, int, int), void *args1) {
  fn = fn1, args = args1;
  int n = num_thread;
  num_pending = n - 1;
  for (int i = 1; i < n; ++i) flags[i] = 1;
  fn(args, 0, n);
  while (num_pending) pthread_yield();
}
