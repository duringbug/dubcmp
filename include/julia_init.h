#ifndef JULIA_INIT_H
#define JULIA_INIT_H

#ifdef __cplusplus
extern "C" {
#endif

void init_julia(int argc, char *argv[]);
void shutdown_julia(int retcode);

#ifdef __cplusplus
}
#endif

#endif // JULIA_INIT_H
