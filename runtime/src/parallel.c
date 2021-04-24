#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <sys/sysinfo.h>

unsigned num_thread;

static atomic_bool flags[128];
static void (*fn)(void *, unsigned);
static void *args;
static atomic_uint num_pending;

_Noreturn static void *thread_fn(void *p) {
  unsigned i = (size_t) p;
  while (1) {
    while (!flags[i]) sched_yield();
    flags[i] = 0;
    fn(args, i);
    --num_pending;
  }
}

void parallel_init(unsigned th) {
  unsigned n = num_thread = (th ? th : (unsigned) get_nprocs()
#if defined(__x86_64__)
    / 2
#endif
  );
  cpu_set_t cpu;
  CPU_ZERO(&cpu);
  for (unsigned i = 0; i < n; ++i) CPU_SET(i, &cpu);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu), &cpu); // master
  for (unsigned i = 1; i < n; ++i) {
    pthread_t th;
    pthread_create(&th, 0, thread_fn, (void *) (size_t) i);
    CPU_ZERO(&cpu);
    CPU_SET(i, &cpu);
    pthread_setaffinity_np(th, sizeof(cpu), &cpu);
  }
}

void parallel_launch(void (*fn1)(void *, unsigned), void *args1) {
  fn = fn1, args = args1;
  unsigned n = num_thread;
  num_pending = n - 1;
  for (unsigned i = 1; i < n; ++i) flags[i] = 1;
  fn(args, 0);
  while (num_pending) sched_yield();
}
